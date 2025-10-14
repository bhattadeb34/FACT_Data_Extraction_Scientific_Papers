from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import base64
import traceback

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from litellm import completion # NEW import

set_tracing_disabled(True)  # avoid extra logging from langchain/tracing

## setup schema for output ##
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Type

class ElectrolyteName(BaseModel):
    """A structured representation of the electrolyte's name."""
    full_name: str = Field(
        ...,
        description="The complete, formal name of the electrolyte or its major components (e.g., Lithium Lanthanum Zirconate Oxide)."
    )
    acronym: Optional[str] = Field(
        None,
        description="The associated abbreviation, chemical formula, or common name (e.g., LLZO, PEO)."
    )
    proportion: Optional[str] = Field(
        None,
        description="The relative stoichiometric ratios, molar ratios, or concentrations of components."
    )

class IonicConductivityDataPoint(BaseModel):
    """Represents a single extracted data point for ionic conductivity."""
    material_class: Literal["Ceramic", "Polymer", "Other"] = Field(
        ...,
        description="The primary functional class of the material."
    )
    electrolyte_name: ElectrolyteName = Field(
        ...,
        description="Structured name of the electrolyte, including its full name, acronym, and component proportions."
    )
    ionic_conductivity_S_per_cm: str = Field(
        ...,
        description="Ionic conductivity value in S cm⁻¹. May include qualitative notes."
    )
    measurement_temperature: str = Field(
        ...,
        description="The temperature at which the measurement was taken (°C, K, or 'RT')."
    )
    specific_source_location: str = Field(
        ...,
        description="Precise location within the source document where the data was found."
    )
    material_description: str = Field(
        ...,
        description="Description of the material's properties, or 'N/A (Cited Work)' if not the primary material of the study."
    )
    processing_method: str = Field(
        ...,
        description="Synthesis and fabrication steps, or 'N/A (Cited Work)' if not the primary material of the study."
    )

class ConductivityData(BaseModel):
    """The root model for the entire JSON output, containing a list of conductivity data points."""
    conductivity_data: List[IonicConductivityDataPoint]

class FigureAnalysis(BaseModel):
    """Analysis of a figure from the paper."""
    figure_type: Literal["data_plot", "schematic", "table", "image", "other", "not_relevant"] = Field(
        ...,
        description="Type of figure: data_plot (contains extractable conductivity data), schematic (conceptual diagram), table (data table), image (photo/SEM), other, or not_relevant (header/footer/logo)"
    )
    contains_conductivity_data: bool = Field(
        ...,
        description="True if the figure contains extractable ionic conductivity data points"
    )
    figure_description: str = Field(
        ...,
        description="Brief description of what the figure shows (e.g., 'Arrhenius plot of LLZO conductivity', 'Schematic of SSB architecture', 'SEM image of pellet')"
    )
    key_message: str = Field(
        ...,
        description="The main takeaway or message from this figure"
    )
    trends_observations: str = Field(
        ...,
        description="Notable trends, patterns, or observations visible in the figure (e.g., 'conductivity increases with temperature', 'LLZO shows higher conductivity than LATP')"
    )
    conductivity_data: List[IonicConductivityDataPoint] = Field(
        default_factory=list,
        description="List of extracted conductivity data points. Empty if figure contains no extractable data."
    )
    comments: str = Field(
        ...,
        description="Additional comments, caveats, or notes about the figure or data extraction quality"
    )

class AggregatedElectrolyteName(BaseModel):
    """A structured representation of aggregated electrolyte names for a group."""
    full_names: List[str] = Field(
        ...,
        description="A list of all unique formal names/major components in the group."
    )
    acronyms: List[str] = Field(
        ...,
        description="A list of all unique acronyms, chemical formulas, or common names in the group."
    )
    proportions: List[str] = Field(
        ...,
        description="A list of all unique proportions, ratios, or specific process types in the group."
    )

class AggregatedConductivityData(BaseModel):
    """Represents a single aggregated row of conductivity data, grouped by material type."""
    grouping_key: str = Field(
        ...,
        description="The primary key for grouping, e.g., 'Garnet-type', 'PEO-based', or 'Other'."
    )
    aggregated_electrolyte_name: AggregatedElectrolyteName = Field(
        ...,
        description="An aggregation of all unique electrolyte names, acronyms, and proportions within the group."
    )
    ionic_conductivity_range_S_per_cm: str = Field(
        ...,
        description="The full range (min - max) of ionic conductivity values observed in the group, in S cm⁻¹."
    )
    measurement_temperature_range: str = Field(
        ...,
        description="The range of measurement temperatures (°C/K) for the group. Can be 'RT' if no specific values are given."
    )
    consolidated_sources: List[str] = Field(
        ...,
        description="A consolidated list of unique source references that contributed data to this group."
    )
    consolidated_material_description: str = Field(
        ...,
        description="A semicolon-separated string of all unique material descriptions for the group. Is 'N/A (Cited Work)' if no primary data exists."
    )
    consolidated_processing_method: str = Field(
        ...,
        description="A semicolon-separated string of all unique processing methods for the group. Is 'N/A (Cited Work)' if no primary data exists."
    )

class AggregatedData(BaseModel):
    """The root model for the aggregated data output, containing a list of grouped data points."""
    aggregated_data: List[AggregatedConductivityData]

# --- ⚙️ Workflow Schema Selector ---

WORKFLOW_SCHEMAS = {
    "extraction": ConductivityData,
    "aggregation": AggregatedData,
    "figure_analysis": FigureAnalysis,
}

def get_workflow_schema(workflow_name: str) -> Type[BaseModel]:
    """
    Retrieves the Pydantic model class for a given workflow name.
    """
    schema = WORKFLOW_SCHEMAS.get(workflow_name)
    if not schema:
        raise ValueError(f"Unknown workflow name: '{workflow_name}'. Available workflows are: {list(WORKFLOW_SCHEMAS.keys())}")
    return schema

## --- ⚙️ PyDantic Encoder ---
def pydantic_encoder(obj):
    """JSON encoder for Pydantic models."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


# --- 🔍 Figure Extraction Functions ---

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_figure_files(file_path: str) -> List[Path]:
    """Find all figure jpeg files in the same directory as the input file."""
    paper_dir = Path(file_path).parent
    # Look for figure patterns
    figures = list(paper_dir.glob("_page_*_Figure_*.jpeg"))
    figures.extend(paper_dir.glob("_page_*_Figure_*.jpg"))
    figures.extend(paper_dir.glob("_page_*_Figure_*.png"))
    return sorted(figures)

async def extract_from_single_figure(
    fig_path: Path,
    model: str,
    api_key: str,
    prompt: str,
    output_schema: Type[BaseModel]
) -> dict:
    """Extract data from a single figure using vision model."""
    print(f"  [info] Processing figure: {fig_path.name}")
    # Read image as base64
    image_base64 = encode_image_to_base64(str(fig_path))

    message_text = f"""Analyze this figure and extract ionic conductivity data if present.
Figure name: {fig_path.name}
[IMAGE DATA PROVIDED BELOW]"""

    try:
        # Use litellm directly instead of agents library for vision
        response = await asyncio.to_thread(
            completion,
            # FIX: Ensure model name has 'gemini/' prefix for Gemini API
            model=f"gemini/{model}" if not model.startswith("gemini/") else model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}\n\n{message_text}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            api_key=api_key,
            response_format={"type": "json_object"}
        )

        # Parse response
        analysis_json = json.loads(response.choices[0].message.content)

        # Validate against schema
        analysis = FigureAnalysis(**analysis_json)

        # Extract data
        figure_type = analysis.figure_type
        contains_data = analysis.contains_conductivity_data
        conductivity_data = analysis.conductivity_data or []

        return {
            'figure': str(fig_path.name),
            'success': True,
            'figure_type': figure_type,
            'contains_data': contains_data,
            'data_points_count': len(conductivity_data),
            'analysis': analysis.model_dump(),
            'error': None
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"    [error] Failed to process {fig_path.name}: {str(e)}")
        print(f"    [debug] Full traceback:\n{error_details}")
        return {
            'figure': str(fig_path.name),
            'success': False,
            'figure_type': 'error',
            'contains_data': False,
            'data_points_count': 0,
            'analysis': None,
            'error': str(e)
        }


async def extract_from_all_figures(
    figures: List[Path],
    model: str,
    api_key: str,
    prompt: str,
    output_schema: Type[BaseModel]
) -> List[dict]:
    """Extract data from all figures."""
    tasks = [
        extract_from_single_figure(fig_path, model, api_key, prompt, output_schema)
        for fig_path in figures
    ]
    results = await asyncio.gather(*tasks)
    return results


# --- 🔄 Main Workflow ---

async def main(
    model: str,
    api_key: str,
    workflow_name: str,
    dependency: str = None,
    file_path: str = './sample_papers/sample1.md',
    instruction: str = None,
    extract_figures: bool = False,
    vision_model: str = "gemini/gemini-2.5-flash",
    output_dir: str = None
):
    # Load prompt
    with open(instruction, 'r') as f:
        prompt_template = f.read()

    output_schema = get_workflow_schema(workflow_name)

    # --- FIGURE EXTRACTION (if requested) ---
    figure_extraction_results = None
    if extract_figures and (workflow_name == "extraction" or workflow_name == "figure_analysis"):
        print("[info] Looking for figures to extract...")
        figures = find_figure_files(file_path)

        if figures:
            print(f"[info] Found {len(figures)} figures")

            # Load figure-specific prompt
            figure_prompt_path = instruction.replace('.txt', '_figures.txt')
            if Path(figure_prompt_path).exists():
                with open(figure_prompt_path, 'r') as f:
                    figure_prompt = f.read()
                print(f"[info] Using figure prompt: {figure_prompt_path}")
            else:
                print(f"[warning] Figure prompt not found at {figure_prompt_path}, using default text prompt")
                figure_prompt = prompt_template

            # Extract from all figures using vision model
            print(f"[info] Using vision model: {vision_model}")
            figure_extraction_results = await extract_from_all_figures(
                figures=figures,
                model=vision_model,
                api_key=api_key,
                prompt=figure_prompt,
                output_schema=output_schema
            )

            # Summary statistics
            successful = sum(1 for r in figure_extraction_results if r['success'])
            with_data = sum(1 for r in figure_extraction_results if r.get('contains_data', False))
            total_data_points = sum(r.get('data_points_count', 0) for r in figure_extraction_results)

            print(f"[info] Figure extraction summary:")
            print(f"  - Total figures processed: {len(figures)}")
            print(f"  - Successfully analyzed: {successful}")
            print(f"  - Figures with conductivity data: {with_data}")
            print(f"  - Total data points extracted from figures: {total_data_points}")

            # Print breakdown by figure type
            type_counts = {}
            for r in figure_extraction_results:
                fig_type = r.get('figure_type', 'unknown')
                type_counts[fig_type] = type_counts.get(fig_type, 0) + 1

            print(f"  - Figure type breakdown:")
            for fig_type, count in sorted(type_counts.items()):
                print(f"    • {fig_type}: {count}")

        else:
            print("[info] No figures found in paper directory")

    # --- TEXT EXTRACTION ---
    # (Skip text extraction if we're only doing figure analysis and it was successful)
    if workflow_name == "figure_analysis" and figure_extraction_results is not None:
        result = None # Placeholder, as the main result is from figures
    else:
        print(f"[info] Running text extraction workflow '{workflow_name}' on file: {file_path}")

        agent = Agent(
            name='Fact LLM Agentic Workflow',
            instructions=prompt_template,
            model=LitellmModel(model=model, api_key=api_key),
            output_type=output_schema
        )

        # Load file content
        with open(file_path, 'r') as f:
            file_content = json.load(f) if file_path.endswith('.json') else f.read()

        if not dependency:
            input_data = file_content
        else:
            input_data = file_content[dependency]
            if not isinstance(input_data, str):
                input_data = str(input_data)

        # Run text extraction
        result = await Runner.run(agent, input_data)

    # --- SAVE RESULTS ---
    # Prepare JSON buffer
    try:
        with open(file_path, 'r') as f:
            file_content_for_buffer = json.load(f) if file_path.endswith('.json') else f.read()
    except FileNotFoundError:
         file_content_for_buffer = ""

    if file_path.endswith('.md'):
        json_buffer = {
            'raw_md': file_content_for_buffer
        }
    elif file_path.endswith('.json'):
        if isinstance(file_content_for_buffer, dict):
            json_buffer = file_content_for_buffer.copy()
        else:
            json_buffer = {}
    else:
        json_buffer = {}

    if result:
         json_buffer[workflow_name] = result.final_output

    if figure_extraction_results:
        if workflow_name == "figure_analysis":
            json_buffer[workflow_name] = figure_extraction_results
        else:
            json_buffer['figure_extraction'] = figure_extraction_results

    # Generate output filename
    prompt_stub = os.path.basename(instruction).replace('.txt', '')
    model_stub = model.replace('/', '-').replace(':', '-')

    # Determine output directory
    if output_dir:
        # If output_dir is provided, mirror the input structure
        input_path = Path(file_path).resolve()
        input_parts = input_path.parts
        try:
            output_idx = input_parts.index('output')
            relative_path = Path(*input_parts[output_idx+1:])
            relative_dir = relative_path.parent
            out_dir = Path(output_dir) / relative_dir
        except (ValueError, IndexError):
            out_dir = Path(output_dir) / input_path.parent.name
    else:
        # Default: same directory as input file
        out_dir = Path(file_path).parent or Path('.')

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    json_path = out_dir / f"{model_stub}_{prompt_stub}_{workflow_name}.json"

    with open(json_path, 'w') as f:
        json.dump(json_buffer, f, indent=4, default=pydantic_encoder)

    print(f"[ok] Wrote output JSON: {json_path}")
    # Wait for 0.1 seconds to ensure the file is written
    await asyncio.sleep(0.1)
    return result
# ContourAgent-backend-python

## Overview
The Kriging Backend Service is a Python-based backend project providing geostatistical interpolation functionalities. It is designed to process spatial data, especially for geological and related fields, generating high-accuracy interpolated surfaces using the Kriging interpolation method, supporting automated and reproducible geospatial analysis workflows.

## Scientific Background and Purpose
Geological contour map generation is a fundamental task in spatial analysis and geoscience research. Conventional workflows require extensive manual parameter configuration and expert knowledge, limiting efficiency and reproducibility. This software aims to reduce manual intervention by providing automated spatial interpolation capabilities.

## Methodology Correspondence
This software implementation corresponds directly to the methodology described in the associated manuscript (if applicable, please replace with actual correspondences):

- Data Preprocessing and Loading: Describe the data input process based on the project's actual situation.
- Geostatistical Modeling and Kriging Interpolation: The `kriging.py` file implements the Kriging interpolation algorithm.
- Result Output and Visualization: Describe the output format of interpolation results or potential visualization methods.

## Software Architecture
This project primarily functions as a Python backend service, typically communicating with frontend applications (e.g., Java frontend or web frontend) via RESTful APIs, using JSON-formatted messages for data exchange.

## Requirements and Dependencies
### Python Backend
- Python 3.8 or later
- All dependencies listed in `requirements.txt` (e.g., `numpy`, `scipy`, `pykrige`, `shapely`, etc.)

## Repository Structure
```
.  
├── api.py                      # Main FastAPI application with Kriging logic and API endpoints
├── nlp_processor.py            # Natural Language Processing related utilities
├── data_query.py               # Implementation of the Data Retrieval Agent
├── image.py                    # Implementation of the Map Generation Agent
├── kriging.py                  # Implementation of the Interpolation Modeling Agent
├── feedback_processor.py       # Implementation of the Feedback Agent
├── prompt.py                   # Prompt related utilities 
├── mcp_server.py               # MCP Server related utilities 
├── mcp_tool.py                 # MCP Tool related utilities 
├── agent.py                    # Implementation of the Multi-Agent system
├── context_schema.py           # Schema definitions for context
├── requirements.txt            # Python dependencies
├── scBasin.geojson             # GeoJSON file defining the basin boundary for masking
└── README.md                   # This README file
```

## Usage and Reproducibility
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Python backend service:
   ```bash
   python api.py
   ```
The backend service will run locally and process spatial data interpolation tasks submitted via the API.

## License
This project is released under the MIT License.

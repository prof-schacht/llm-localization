# Project Status: Moral Foundation Analysis in Language Models

## Session: [Date]

### 1. Model Configuration and Setup
- Added support for Microsoft's Phi-4 model in `model_utils.py`
  - Configured 40 layers and 5120 hidden dimensions
  - Updated model name mappings to handle different variants of the model name
  - Implemented layer name generation for Phi architecture

### 2. Moral Foundation Analysis
- Implemented localization script to analyze moral foundations in Phi-4
- Created analysis pipeline for different moral foundations:
  - care
  - fairness
  - loyalty
  - authority
  - sanctity
  - liberty
- Set up batch processing with `localize_example.sh` to analyze all foundations

### 3. Visualization and Analysis Tools
- Developed `analyze_mask.py` for visualizing unit distributions
  - Created heatmap visualizations of active units across layers
  - Added statistical analysis capabilities
  - Implemented percentage-based unit selection
- Enhanced visualization with:
  - Layer-wise distribution plots
  - Percentage annotations
  - Customizable plot parameters

### 4. Automated Analysis Pipeline
- Created automated analysis script (`analyze_mask.sh`)
  - Processes all .npy files in cache directory
  - Extracts parameters from filenames
  - Generates standardized plots for each analysis
  - Saves results in organized plot directory

### 5. Technical Improvements
- Fixed device mapping issues for Phi-4 model
- Implemented error handling and parameter validation
- Added support for different pooling strategies:
  - last-token
  - mean pooling

### 6. Results and Observations
- Successfully generated distribution plots for different moral foundations
- Created plots showing unit activation patterns across model layers
- Stored results in:
  - `cache/` directory for masks
  - `plots/` directory for visualizations

### Next Steps
1. Analyze differences in unit distributions across moral foundations
2. Compare results with other model architectures
3. Investigate correlation between moral foundation units and model behavior
4. Consider expanding analysis to other ethical dimensions

### Technical Notes
- Model specifications:
  - Architecture: Phi-4
  - Layers: 40
  - Hidden dimension: 5120
  - Analysis percentage: 1% and 5% of units
- Analysis parameters:
  - Pooling: mean and last-token
  - Range: 100-100 (top selective units)
  - Foundation-specific analysis 

----------
# Project Status: Moral Foundation Analysis in Language Models

## Session: [Current Date]

### Statistical Analysis Improvements

#### 1. Enhanced Visualization Implementation
- Added statistical visualization capabilities to analyze unit selection
- Created new plots showing:
  - T-value distributions across layers
  - -log10(p-value) distributions across layers
  - Visual indicators for significance thresholds
  - Selection threshold indicators for percentage-based cutoffs

#### 2. Key Findings from Statistical Analysis
- Observed that moral foundation detection (e.g., "care") shows:
  - Bidirectional effects (both positive and negative t-values)
  - Distribution across all layers rather than localization
  - Many statistically significant units (p < 0.05)
  - T-values ranging approximately from -10 to +10
  - High statistical significance (-log10(p-values) often > 10, indicating p < 1e-10)

#### 3. Unit Selection Methodology Discussion
- Current approach:
  1. Compute t-values and p-values for ALL units
  2. Plot full distribution of statistics
  3. Select top 1% based on absolute t-values
- Findings suggest 1% selection might be too conservative:
  - Many statistically significant units are excluded
  - Effects appear distributed across layers
  - High number of units show strong statistical significance

#### 4. Proposed Improvements
Identified potential alternative selection criteria:
1. Higher percentage threshold
2. Statistical significance-based selection
3. Hybrid approach combining:
   - Statistical significance (p-value threshold)
   - Effect size (t-value magnitude)
   - Percentage-based limits

### Next Steps
1. Evaluate alternative unit selection criteria
2. Compare unit selection patterns across different moral foundations
3. Consider implementing adaptive thresholds based on foundation-specific patterns
4. Analyze relationship between statistical significance and model behavior

### Technical Notes
- Added visualization of selection thresholds
- Implemented statistical summary metrics
- Enhanced documentation of selection methodology
- Created foundation-specific statistical distribution plots

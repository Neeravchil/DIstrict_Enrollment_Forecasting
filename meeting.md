
All projects
Enrollment Projection



How can I help you today?


Analyzing new enrollment trends for ML prediction
Last message 2 days ago
School enrollment data by grade and month
Last message 3 days ago
Using table for monthly enrollments
Last message 3 days ago
Timesheet description for 2 points
Last message May 28
Quick model development guide for new data scientist
Last message May 27
Memory
Only you
Purpose & context Rohit is a data scientist based in India, currently working on a Chicago Public Schools (CPS) enrollment forecasting project. The project involves building ML solutions for predicting student enrollment, with a strong emphasis on delivering quick, working results (roughly one-week sprint cycles) rather than extended development. Key stakeholder on the CPS side is a contact in CPS's Planning and Demographics team. Rohit is relatively new to this project and the CPS domain, and has been building context rapidly through documentation and onboarding materials. Current state Active work spans several interconnected streams: Streamlit dashboard development: Refactoring prediction pipelines, implementing dynamic filtering and subgroup selectors, resolving session state issues, and optimizing performance with caching. Dashboard has been deployed to a staging environment. Meal forecasting feature: Testing and evaluation using holdout windows and error metrics (MAE, RMSE), comparing aggregation strategies, and navigating small-school volatility as a modeling challenge. Absenteeism analysis: Identified early-year absence rates (Days 1–20) as a leading indicator correlated with end-of-year enrollment drop-off — a quick-win analytical finding. Rohit has also produced domain knowledge documentation to get up to speed on how the Chicago school system operates, including school types, grade structures, the student assignment/choice process, and enrollment trend drivers. On the horizon Continued iteration on the ML forecasting model, including handling data anomalies (notably the 2022–2024 migrant student enrollment spike and its 2025–26 reversal) Closing knowledge gaps in CPS domain understanding by verifying low-confidence facts with the team Progressing from staging to production on the Streamlit dashboard Key learnings & principles The 2022–2024 migrant enrollment anomaly is a flagged data issue requiring explicit handling in any model trained on historical data Small-school enrollment volatility is a known modeling challenge that affects forecasting reliability Early-year absenteeism is a practically useful leading indicator worth surfacing to stakeholders Given Rohit's remote location relative to the project, confidence-scoring factual claims and flagging uncertainties explicitly is a useful practice for team coordination Approach & patterns Prefers direct, jargon-free communication for conceptual understanding, but uses technically specific, credibility-signaling language for professional documentation (e.g., timesheets, reports shared with stakeholders) Favors quick wins and practical output over theoretical depth Works iteratively with a sprint mindset, prioritizing a working solution quickly over a perfect one Uses structured documents (Word/.docx format) for onboarding and knowledge transfer artifacts Tools & resources Streamlit for dashboard development Python ML stack (implied by pipeline/cache patterns) Error metrics: MAE, RMSE for model evaluation Word documents (.docx) for deliverable artifacts and documentation

Last updated yesterday

Instructions
Add instructions to tailor Claude’s responses

Files
1% of project capacity used

First Meeting with Ileana
324 lines

text


Project Proposal for K12 School District Enrollment Projections.pdf
pdf


CPS School Enrollment Projections Documentation.pdf
pdf


First Meeting with Ileana
# Project Context — CPS Enrollment Forecasting ML System
 
## Overview
This project is about improving student enrollment forecasting for CPS (Chicago Public Schools) using machine learning.
 
The current forecasting system already exists and is implemented primarily in SAS using manually coded logic. The team is exploring whether machine learning can improve forecasting accuracy, especially for entry-level grades such as Kindergarten and 9th grade.
 
The system is operationally important because enrollment forecasts influence:
- staffing
- budgeting
- lunch planning
- school space planning
- overcrowding management
- under-enrollment decisions
 
---
 
# Key Stakeholders
 
## Eliana Vargas
Role:
- Manages Planning and Demographics team at CPS.
 
Team responsibilities:
- Enrollment analysis
- Spatial/geographic analysis
- Attendance boundaries
- School capacity analysis
- Overcrowding / under-enrollment studies
- Enrollment projections
 
Eliana is currently one of the primary people maintaining the existing enrollment projection system.
 
---
 
# Current Forecasting System
 
## Current Methodology
CPS currently forecasts enrollment using Cohort Survival Rates.
 
### Cohort Survival Concept
Example:
- Previous year Grade 6 enrollment
- Current year Grade 7 enrollment
 
Formula:
 
Survival Rate =
(Current Year Grade 7 Enrollment)
/
(Previous Year Grade 6 Enrollment)
 
Interpretation:
- 1 → no enrollment change
- >1 → enrollment increase
- <1 → enrollment decline
 
This is calculated:
- per school
- per grade transition
- using historical data
 
---
 
# Existing System Characteristics
 
The current system:
- is implemented in SAS
- uses approximately 10 years of historical data
- automatically selects averaging windows
- attempts to minimize historical forecasting error
- is heavily manual and difficult to maintain
 
The system uses:
- averages
- historical cohort survival rates
- retrospective evaluation logic
 
Eliana mentioned that there are “some aspects of machine learning” in the current process, but it is mostly statistical rule-based logic.
 
---
 
# Main Business Problems Identified
 
## 1. Entry-Level Grade Forecast Errors
Highest forecast errors occur in:
- Kindergarten
- 9th Grade
 
Reason:
The current system relies heavily on historical averages.
 
Problem:
District enrollment is declining overall, so historical averages often overestimate future enrollment.
 
This creates compounding downstream forecasting errors because future grades depend on earlier grade projections.
 
---
 
## 2. School Choice Behavior Not Properly Modeled
Students do not always attend their geographically assigned school.
 
Current system does not adequately model:
- school choice patterns
- lottery-based enrollment
- zoned vs non-zoned attendance
 
This is especially important for 9th grade forecasting.
 
---
 
## 3. Small Schools Have High Variability
Many schools have small enrollment sizes, causing:
- unstable enrollment trends
- fluctuating survival rates
- high forecasting volatility
 
---
 
## 4. Special Programs Affect Enrollment
Certain schools contain:
- special education programs
- English learner programs
- other special enrollment programs
 
These impact enrollment patterns.
 
---
 
# Attendance Boundary Context
 
CPS contains geographically assigned schools.
 
Students are assigned a “zoned school” based on where they live.
 
However:
- students may apply to other schools
- some schools admit non-zoned students through lottery/application systems
 
Therefore schools may contain:
- zoned students
- non-zoned students
 
This distinction is operationally important.
 
---
 
# Data Currently Available
 
## Enrollment Data
Available:
- 10+ years officially
- possibly 20+ years historically
 
Granularity:
- student-level data exists
 
Includes:
- school attended
- grade
- geography/location
- race/ethnicity
- enrollment indicators
 
---
 
## Geography / Zoning Data
Available:
- student location data
- zoned school information
- attendance boundary relationships
 
Not all zoning-related fields are currently standardized in the data warehouse.
 
---
 
## School Choice Data
CPS has:
- student choice pattern data
- enrollment movement behavior
 
This is considered an important missing signal in current forecasting.
 
---
 
## Birth Data
Available:
- city-level birth counts
 
Used currently for:
- estimating future Kindergarten enrollment
 
Limitation:
- no neighborhood-level birth geography currently available
 
---
 
# Current Forecasting Scope
 
The primary operational forecasting need is:
 
Forecast:
- school-level enrollment
- grade-level enrollment
- for the beginning of the next school year
 
Forecasting is primarily annual.
 
They are NOT primarily interested in:
- monthly enrollment forecasts
 
---
 
# Existing Technical Environment
 
## Current
- SAS
- manual projection logic
 
## Future Direction
District is transitioning toward:
- Python-based workflows
 
---
 
# Proposed Technical Direction
 
Potential stack discussed:
- Microsoft Fabric
- Python
- ML/statistical libraries
- train/test evaluation pipelines
 
Possible deployment:
- model packaging
- operational deployment for district use
 
No final architecture has been decided yet.
 
---
 
# Potential Modeling Directions Discussed
 
Two broad approaches were discussed:
 
## 1. Machine Learning
Using many features:
- enrollment history
- zoning
- demographics
- choice behavior
- geography
- births
 
to predict next-year enrollment.
 
---
 
## 2. Time Series Forecasting
Using:
- trends
- seasonality
- historical patterns
- anomalies
 
to forecast future enrollment.
 
No final modeling choice has been made.
 
---
 
# Potential Long-Term Expansion Ideas Mentioned
 
Not currently implemented deeply, but discussed as future possibilities:
- migration analysis
- neighborhood trend analysis
- gentrification analysis
- school closure planning
- new school planning
- capacity planning
 
---
 
# Important Operational Constraints
 
## Manual Bottleneck
Eliana currently handles much of the forecasting logic herself.
 
Current system updates:
- are slow
- require manual effort
- are difficult to maintain
 
Bandwidth constraints exist.
 
---
 
# Project Expectations
 
Initial goals:
- understand current methodology
- gather documentation
- identify available datasets
- define inputs and outputs
- compare ML approach against current baseline
 
Initial work expected:
- proof of concept
- prototype / draft system
- comparison against current forecasting approach
 
---
 
# Important Unknowns / Not Yet Confirmed
 
The following are NOT finalized:
- final ML model choice
- exact target metric
- deployment architecture
- complete data schema
- data cleanliness
- complete historical forecast-vs-actual availability
- production workflow
- evaluation methodology

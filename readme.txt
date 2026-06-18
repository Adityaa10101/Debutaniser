Objectives 
  “AI Based Model to Minimize C4 Slippage in DEBUTANIZER” 

Problem Statement
  C4 slippage in C5+ product stream varies 0.8 % to 1.5% (Spec 0.5 M%)
  Manual operation based on experience 
  Analyzer cycle time 12 minutes (Wide variation between sampling)
 Analyzer reading not reliable
  Feed and operating variability not handled optimally 








Input variables 
  Top & Bottom temperature 
  Reboiler outlet temperature 
  Reboiler steam flow
  Reflux flow
  Feed flow
 Control tray temperature 
 Tray Temperature
 Bottom Analyzer
 Re-boiling steam flow

Soft Sensor Model
  Model : Gradient Boosting Regression (Based on the Sof sensor prediction column process parameter will be optimized accordingly)
  Feature Engineering : Steam flow, Reflux ratio, Temp. diff, feed flow
  Output : Predicted C4 wt.% in DB bottom 

Dashboard Design
  Live C4 prediction 
  Actual vs predicted trends for C4 
  Operator recommendations with loss INR/hr calculations

Optimization Strategy 
  Dynamic adjustment of reflux and steam 
  Maintain optimal bottom temperature 
  Balance energy vs recovery

Future Scope
 Real time deployment in Seeq/other options to be explored.
 Closed loop optimization with APC. 
 Extend to other columns. 


Solution architecture 
  Tailing tower data from Exaquantum and lab data.
  AI based soft sensor for C4 prediction
  Real time optimization along with operator instructions.



Process Details:
To separate mixed C4s from C5s and heavier. 
DP bottom is fed on level control to the 17th tray of debutanizer
Reboiling duty is provided by LP (desuperheater) steam.
Column vapors are condensed with cooling water and collected in reflux drum.
Mixed C4s after meeting reflux requirement are sent for further processing to: 
            Butadiene Extraction Unit.
            C4 hydrogenation Unit. 
            OSBL Storage.




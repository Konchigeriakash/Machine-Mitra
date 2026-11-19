# MachineMitra_Project_Details

# MachineMitra

A low-cost sensor that warns small factories before their machines break down

|Item|Detail|
|---|---|
|Event|Tech4Bharat, GAVS Hackathon (HackCulture)|
|Theme|Industrial Manufacturing|
|Problem statement|4.3 Small factories lose production to breakdowns they cannot see coming|
|Team lead|Vinut Vishwanath Dambal (ECE student)|
|Idea submission deadline|15 Nov 2026 (shortlist announced 23 Nov, finale 27 Dec 2026, Bangalore Palace)|

## 1. The Problem

Powerlooms, flour mills, pump sets and small machine shops run old machines with no condition monitoring and no maintenance team. The owner also acts as the maintenance department and judges machine health by sound and feel. Machines run until they fail. Then the owner calls whichever mechanic is free and sends someone to town for the part. This costs lost production days, missed deliveries, emergency prices for parts and wages paid to idle workers.

## 2. Our Idea

A small clip-on sensor node on each machine learns how that machine normally runs. When the vibration pattern starts to drift, it warns the owner early. A simple agent then turns the warning into a plan: which machine to fix first, the best idle time for repair, and a ready message to the mechanic with the likely part.

## 3. How the System Works

|Step|What happens|
|---|---|
|1. Sense|ESP32 with a vibration sensor (MPU6050) reads the machine. Optional: current and temperature<br>sensors.|
|2. Learn normal|For 1-2 weeks the node learns what normal running looks like. No fault labels are needed.|
|3. Detect|Features (RMS, peak, FFT frequencies) go to an anomaly detector. It gives a health score.|
|4. Work offline|The model runs on the ESP32. Data is stored on the device and alerts go out when the network<br>returns.|
|5. Agent plans|Ranks machines by risk, picks a repair time in an idle window, drafts a message to the mechanic<br>with the likely part.|
|6. Alert|Message on Telegram or WhatsApp in Kannada or Hindi, simple words.|
|7. Log|Every repair is saved, so the system gets better over time.|

## 4. Hardware Needed

|Part|Use|Approx. cost (Rs.)|
|---|---|---|
|ESP32 board|Brain, Wi-Fi, runs the model|400-600|
|MPU6050 or ADXL345|Vibration sensor|100-250|
|ACS712 or SCT-013 (optional)|Motor current|150-400|
|DS18B20 or LM35 (optional)|Motor temperature|50-150|
|Small DC motor or fan|Test machine for the demo|200-500|
|Power supply, wires, breadboard, case|Setup|300-500|
|Total||about 1,200-2,500|

|Power supply, wires, breadboard, case<br>Setup|||||
|---|---|---|---|---|
|Total|||||
|Costs are rough estimates. If you buy only two thi|ngs first, buy|the ESP32|and the|MPU6050.|
|5. Software Needed|||||
|Tool|Use||||
|Arduino IDE or PlatformIO|Program the|ESP32|||
|Python (NumPy, SciPy, scikit-learn)|Process data|, build and|test the|anomaly d|
|MATLAB / Simulink (optional)|Signal analys|is, same a|s the fall|detection p|
|Telegram bot or WhatsApp|Send alerts t|o the owne|r||
|Simple phone screen|Show health|score and|machine|ranking|

|Tool|Use|
|---|---|
|Arduino IDE or PlatformIO|Program the ESP32|
|Python (NumPy, SciPy, scikit-learn)|Process data, build and test the anomaly detector|
|MATLAB / Simulink (optional)|Signal analysis, same as the fall detection project|
|Telegram bot or WhatsApp|Send alerts to the owner|
|Simple phone screen|Show health score and machine ranking|

<!-- Page 3 -->

## 6. Scope: What Is In and What Is Out

|In the project|Not in the project (and why)|
|---|---|
|Vibration-based anomaly detection learned from normal<br>running|Fault-type classifier: there is no labelled fault data|
|Offline working on the ESP32|Automatic machine stopping: the owner decides to stop<br>production|
|Agent that ranks risk and suggests repair time|A fancy app: a chat message is enough for these users|
|Alerts in Kannada or Hindi||
|Repair log||

## 7. Data Plan

-**Public datasets (for early testing):**CWRU Bearing, NASA IMS Bearing, MaFaulDa. Use them to build and test the code while the parts arrive. Also a backup if hardware is late, since the problem allows recorded or simulated signals.

-**Own recordings (for the real demo):**record 15-30 minutes of a fan or motor running normally, then record again with a small fault such as a weight on the blade or a loose mount.

## 8. Demo Plan

- Attach the sensor to a fan or motor.
- Run it normally and show a steady health score.
- Add a small weight to the blade and show the score drop.
- Show the alert message and the repair suggestion with the machine ranking.

## 9. Who Benefits and How It Can Earn

|Who|Benefit|
|---|---|
|Mill and powerloom owners|Fewer surprise breakdowns, less idle wage cost, planned repairs|
|Local mechanics|Know which machine needs work and plan their week|
|Spare part dealers|Advance notice of parts needed|

**Possible earning model (to be confirmed with real owners):**a low one-time kit price plus a small monthly fee for alerts and repair planning, or a service deal with local mechanics who install and maintain the kits.

Final prices should come from field answers, not guesses.

## 10. Questions to Ask a Real Owner

- Which machine failed last, how long was it down, and what did it cost?
- What did workers notice before it failed?
- How long does it take to find a technician and a part?
- Are there idle windows when repairs could happen without losing output?
- What is the power and phone signal like on the shop floor?
- What record of past repairs exists?

Even one or two real answers will make the submission stronger.

## 11. Timeline to 15 Nov

|When|Task|Output|
|---|---|---|
|29-31 Oct|Order ESP32 and MPU6050. Download a public vibration dataset.|Parts ordered, data ready|
|1-5 Nov|Write the idea submission: problem, solution, block diagram, agent flow,<br>cost, earning model.|Submission draft|
|1-8 Nov|Wire the sensor, record normal and faulty vibration on a fan or motor.|Real data|
|6-12 Nov|Anomaly detection in Python, then a simple version on the ESP32.|Working detector|
|10-14 Nov|Telegram alert flow and risk-ranking screen for 2-3 machines.|Demo|
|By 15 Nov|Final check and submit.|Done|

|1-5 No|v|Write the idea|submission: proble|m, sol|ution, block diagra|m, agent|flow,|Submission|
|---|---|---|---|---|---|---|---|---|
|||cost, earning m|odel.||||||
|1-8 No|v|Wire the senso|r, record normal a|nd faul|ty vibration on a f|an or mot|or.|Real data|
|6-12 N|ov|Anomaly detec|tion in Python, the|n a sim|ple version on th|e ESP32.||Working de|
|10-14|Nov|Telegram alert|flow and risk-rank|ing scr|een for 2-3 machi|nes.||Demo|
|By 15|Nov|Final check an|d submit.|||||Done|
|Other|deadlines|overlap: Skillinf|round (1-4 Nov) a|nd the|Bits2Wave PPT (|10 Nov).|Writing c|omes first a|
|runs in|parallel.||||||||
|12.|Risks a|nd Fixes|||||||
|Risk||||Fix|||||
|Parts|arrive late|||Use pu|blic datasets and|recorded|signals|first|
|Event|focus is rur|al and small b|usiness|Pitch it|for powerlooms, f|lour mills|and sm|all workshop|
|Too lit|tle time for|the ESP32 mo|del|Show P|ython detection fi|rst, then|port a si|mple versio|
|Little t|ime becaus|e of other eve|nts|Finish t|he written submis|sion first,|build th|e demo afte|

|Risk|Fix|
|---|---|
|Parts arrive late|Use public datasets and recorded signals first|
|Event focus is rural and small business|Pitch it for powerlooms, flour mills and small workshops|
|Too little time for the ESP32 model|Show Python detection first, then port a simple version|
|Little time because of other events|Finish the written submission first, build the demo after|

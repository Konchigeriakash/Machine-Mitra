# MachineMitra_Team_Plan

# MachineMitra: Team Plan

Who does what, how we work from two different colleges, and the full software brief

|Item|Detail|
|---|---|
|Project|Tech4Bharat (GAVS Hackathon), Industrial Manufacturing, problem 4.3: small factories<br>lose production to breakdowns they cannot see coming|
|Idea in one line|A low-cost sensor learns how a machine normally runs, warns before it breaks, and an<br>agent tells the owner what to fix and when|
|Team|2 members from different colleges, working remotely|
|Idea submission deadline|15 Nov 2026|

## 1. Who Does What

|Hardware person|Software person|
|---|---|
|Buys parts, wires the ESP32 and MPU6050|Loads public datasets and builds the feature code|
|Records normal and faulty vibration on a fan or motor|Builds the anomaly detector and health score|
|Saves recordings as CSV and uploads to the shared Drive|Tests the detector on the shared recordings|
|Runs the final detector on the ESP32|Builds the agent: risk ranking, repair time, mechanic<br>message|
|Records the demo video|Builds the Telegram alert and the repair log|
|Asks a local mill or workshop owner the field questions|Sends the final model or thresholds to the hardware<br>person|

Both together: write the idea submission, final testing, and the pitch.

## 2. How We Work From Different Places

|Need|Tool|How we use it|
|---|---|---|
|Share code|GitHub (private repo)|Both push and pull. Work in separate files to avoid clashes.|
|Write and run Python<br>together|Google Colab|One shared notebook, like a Google Doc. Runs in the<br>browser, no setup.|
|Share data and files|Google Drive|One shared folder: data, code, docs.|
|Talk and share screen|Google Meet or WhatsApp<br>video|Short call every 2-3 days. Share screen to debug.|
|Write the submission|Google Docs|Edit the same draft live.|

## Shared folder layout

MachineMitra/ data_public/     CWRU and NASA IMS files data_ours/       recordings from the hardware person (CSV) code/            notebooks and scripts docs/            submission draft, PDFs, diagrams

## The handoff agreement

|Item|Agreement|
|---|---|
|File format|CSV with columns: time, x, y, z|

<!-- Page 2 -->

|Item|Agreement|
|---|---|
|Sampling rate|Fixed and written in the file name, for example fan_normal_500hz_01.csv. The hardware<br>person confirms the real rate achieved.|
|Files to send|Normal runs (15-30 min in total) and faulty runs (5-10 min), uploaded to data_ours|
|Fault used for the demo|A small weight on the fan blade, or a loose mount. Write it in the file name.|
|Final model|Software person sends thresholds or a simple model. Hardware person runs it on the<br>ESP32.|

<!-- Page 3 -->

## 3. Hardware Person: Full Responsibility

|Part|Use|Approx. cost (Rs.)|
|---|---|---|
|ESP32 DevKit V1 (30 pin)|Brain and Wi-Fi|400-600|
|MPU6050 module (GY-521)|Vibration sensor|100-200|
|Half-size breadboard|Wiring|60-100|
|Jumper wires|Wiring|50-100|
|USB data cable|Programming the ESP32 (must be a data<br>cable)|80-150|
|Table fan or small motor|Test machine|Free if available|

|MPU6050 pin|ESP32 pin|
|---|---|
|VCC|3.3V|
|GND|GND|
|SDA|GPIO 21|
|SCL|GPIO 22|

Use 3.3V, not 5V. Leave AD0 and INT unconnected. Install the Adafruit MPU6050 library in Arduino IDE.

## Hardware steps

- Wire the sensor and check readings in the Arduino Serial Plotter at 115200 baud.
- Fix the sensor firmly on the fan or motor body. A loose sensor gives bad data.
- Switch from the simple test code to a fixed sampling rate and log to CSV (time, x, y, z).
- Record normal running, then record again with the small fault. Keep the sensor in the same place for both.
- Upload the CSV files to the shared Drive with clear file names.
- When the software person sends the model or thresholds, run it on the ESP32 and test live.
- Record the demo video: normal run, add the fault, show the health score drop and the alert.

## 4. Software Brief (for the software person)

**Goal:**turn vibration data into a health score, then into a plan and a message for the owner.

|#|Module|What to build|Tools|
|---|---|---|---|
|1|Data loading|Load CWRU and NASA IMS. Cut the signal into windows (for<br>example 1 second).|Python, pandas, scipy|
|2|Features|Per window: RMS, peak, crest factor, kurtosis, FFT peaks<br>and band energy.|numpy, scipy|
|3|Anomaly<br>detector|Train on normal windows only. Score new windows. Higher<br>score means more abnormal.|scikit-learn (Isolation Forest) or<br>mean and standard deviation<br>of features|
|4|Health score|Convert the score to 0-100 and a status: OK, Watch, Service<br>soon.|Python|
|5|Early warning<br>test|On NASA IMS, show the score rising before failure.|matplotlib|
|6|Agent layer|Take health scores of 2-3 machines. Rank by risk, pick a<br>repair time from idle windows, draft the mechanic message<br>and likely part.|Python rules first, an LLM later<br>if time allows|
|7|Alerts|Send the message to Telegram in Kannada or Hindi, in<br>simple words.|Telegram bot, requests|
|8|Repair log|Save every repair and alert in a CSV file.|pandas|

|7|Alerts||simple words.||Send the message to Telegram in Kannada or Hindi, in||Telegram bot, requests|
|---|---|---|---|---|---|---|---|
|8|Repair log|Rules for the software person||Save every repair and alert in a CSV file.||pandas||

- Train on normal data only. Real factories have no labelled fault data. Faulty data is only for checking that

the score goes up.

- Public datasets are recorded at a much higher sampling rate than the MPU6050. Do not hardcode the

sampling rate. Build the code to work with any rate.

- Keep the detector simple (feature thresholds or a small statistical model) so it can run on the ESP32. A

stronger model can stay on the laptop for comparison.

- The system only advises. It never stops a machine, because the owner decides.

## Software and datasets to install or download

|Item|Use|Source|
|---|---|---|
|Python 3.10 or newer|Main language|python.org|
|Google Colab|Shared notebooks in the browser|colab.research.google.com|
|VS Code (optional)|Offline editor|code.visualstudio.com|
|Git and GitHub|Share code|git-scm.com, github.com|
|Arduino IDE (hardware<br>person)|Program the ESP32|arduino.cc|
|Telegram|Alert bot|telegram.org|
|CWRU Bearing data|Normal and faulty bearing vibration|engineering.case.edu/bearingdatacenter/dow<br>nload-data-file|
|NASA IMS Bearing data|Run-to-failure data|Search: NASA Prognostics Center of<br>Excellence Data Set Repository, then<br>Bearing Data Set|
|MaFaulDa (optional)|Motor faults such as imbalance|Search the dataset name on its official page|

Python libraries: pip install numpy scipy pandas scikit-learn matplotlib (already installed in Colab). Dataset pages sometimes move, so search the dataset name if a link is dead.

## 5. Suggested Order and Timeline

|When|Hardware person|Software person|
|---|---|---|
|29-31 Oct|Order ESP32 and MPU6050|Set up Colab, GitHub, Drive. Download CWRU.|
|1-5 Nov|Wire and test the sensor. Start recording.|Modules 1-3 on CWRU. Both: write the<br>submission.|
|6-10 Nov|Record normal and faulty data, upload CSV|Module 5 on NASA IMS, module 4, test on shared<br>recordings|
|10-14 Nov|Port the detector to the ESP32|Agent layer and Telegram alerts (modules 6-8)|
|By 15 Nov|Demo video|Final check. Both submit.|

The written submission is due first and does not need the hardware, so finish it early. Public datasets let the software

work start before the parts arrive.

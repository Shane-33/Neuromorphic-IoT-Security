% To run this script, you must download the data published at
% https://zenodo.org/records/12734987
% Nayid Triana-Guzman - Jul 2023

clear

sourcePath   = 'C:\Users\natri\OpenNEURO\'; % can be a Windows path
targetPath   = 'C:\Users\natri\OpenNEURO\BIDS_EXPORT\';
nSubject     = 32; % number of subjects to export

if exist(targetPath), rmdir(targetPath, 's'); end

% raw data files
% ----------------------------------
allmyfiles = dir(fullfile(sourcePath, 'raw_data\', '*.set'));
allmyfilenames = string({allmyfiles.name})';

for n = 1:size(allmyfilenames,1)
   data(n).file = {fullfile(sourcePath, 'raw_data\', allmyfiles(n).name)};
end

% general information for dataset_description.json file
% -----------------------------------------------------
generalInfo.Name = 'EEG data offline and online during motor imagery for standing and sitting';
generalInfo.ReferencesAndLinks = { 'Triana-Guzman N, Orjuela-Cañon AD, Jutinico AL, Mendoza-Montoya O and Antelis JM (2022) Decoding EEG rhythms offline and online during motor imagery for standing and sitting based on a brain-computer interface. Front. Neuroinform. 16:961089. https://doi.org/10.3389/fninf.2022.961089' };
generalInfo.BIDSVersion = '1.8.0';
generalInfo.HEDVersion = '8.1.0';
generalInfo.License = 'CC0';
generalInfo.Authors = {'Nayid Triana-Guzman', 'Alvaro D Orjuela-Cañon', 'Andres L Jutinico', 'Omar Mendoza-Montoya', 'Javier M Antelis' };
generalInfo.DatasetDOI = 'doi:10.18112/openneuro.ds005342.v1.0.3';
generalInfo.Acknowledgements = 'We would like to thank the Ministerio de Ciencia, Tecnología e Innovación of Colombia-Minciencias/Colciencias, Universidad Antonio Nariño (UAN), Universidad del Rosario, and Tecnologico de Monterrey for all their support in the development of this dataset';
generalInfo.HowToAcknowledge = 'Triana-Guzman N, Orjuela-Cañon AD, Jutinico AL, Mendoza-Montoya O and Antelis JM (2024). EEG data offline and online during motor imagery for standing and sitting. OpenNeuro Dataset ds005342. doi: doi:10.18112/openneuro.ds005342.v1.0.3';
generalInfo.Funding = 'This research was funded by Ministerio de Ciencia, Tecnología e Innovación of Colombia-Minciencias/Colciencias, contract 594-2019, project 80740-594-2019, and in part by the University of Illinois Chicago and Tecnologico de Monterrey (UIC-TEC) Seed funding Program 2021-2022';
generalInfo.EthicsApprovals = 'The experimental protocol was approved by the ethics committee of the Universidad Antonio Nariño';

% participant information for participants.tsv file
% -------------------------------------------------
pInfo = { 'gender'   'age'   'handedness'   'major';
'F'	22	'right_handed'	'nursing';
'F'	27	'right_handed'	'nursing';
'F'	21	'right_handed'	'nursing';
'M'	22	'right_handed'	'medicine';
'M'	26	'right_handed'	'industrial_automation';
'M'	21	'right_handed'	'nursing';
'M'	27	'right_handed'	'medicine';
'F'	20	'right_handed'	'nursing';
'F'	20	'right_handed'	'nursing';
'M'	22	'right_handed'	'medicine';
'M'	22	'right_handed'	'medicine';
'F'	19	'left_handed'	'nursing';
'M'	20	'right_handed'	'medicine';
'F'	22	'right_handed'	'medicine';
'M'	28	'right_handed'	'medicine';
'M'	23	'right_handed'	'medicine';
'M'	24	'right_handed'	'medicine';
'F'	22	'right_handed'	'nursing';
'F'	24	'right_handed'	'medicine';
'F'	22	'left_handed'	'medicine';
'F'	21	'right_handed'	'medicine';
'F'	25	'right_handed'	'medicine';
'M'	22	'right_handed'	'medicine';
'M'	29	'right_handed'	'medicine';
'F'	24	'right_handed'	'industrial_engineering';
'F'	25	'right_handed'	'medicine';
'F'	23	'right_handed'	'medicine';
'F'	19	'right_handed'	'medicine';
'M'	22	'right_handed'	'medicine';
'M'	22	'left_handed'	'medicine';
'M'	23	'right_handed'	'medicine';
'M'	27	'right_handed'	'medicine'};

% participant column description for participants.json file
% ---------------------------------------------------------
pInfoDesc.gender.Description = 'sex of the participant';
pInfoDesc.gender.Levels.M = 'biological male';
pInfoDesc.gender.Levels.F = 'biological female';
pInfoDesc.participant_id.LongName = 'participant identifier';
pInfoDesc.participant_id.Description = 'unique participant label';
pInfoDesc.age.Description = 'age of the participant';
pInfoDesc.age.Units       = 'years';
pInfoDesc.handedness.Description = 'dominant hand of the participant';
pInfoDesc.handedness.Levels.right_handed = 'righty';
pInfoDesc.handedness.Levels.left_handed = 'lefty';
pInfoDesc.major.Description = 'undergraduate participant';
pInfoDesc.major.Levels.nursing = 'undergraduate student majoring in nursing';
pInfoDesc.major.Levels.medicine = 'undergraduate student majoring in medicine';
pInfoDesc.major.Levels.industrial_engineering = 'undergraduate student majoring in industrial engineering';
pInfoDesc.major.Levels.industrial_automation = 'undergraduate student majoring in industrial automation';

% event column description for events.json file
% ----------------------------------------------------------------------
eInfoDesc.onset.LongName = 'event onset';
eInfoDesc.onset.Description = 'onset (in seconds) of the event measured from the beginning of the acquisition of the first volume in the corresponding task imaging data file';
eInfoDesc.onset.Units = 'second';
eInfoDesc.duration.LongName = 'event duration';
eInfoDesc.duration.Description = 'duration of the event (measured from onset) in seconds';
eInfoDesc.duration.Units = 'second';
eInfoDesc.sample.LongName = 'event sample';
eInfoDesc.sample.Description = 'onset of the event according to the sampling scheme of the recorded modality (i.e., referring to the raw data file that the events.tsv file accompanies)';
eInfoDesc.sample.Units = 'event sample starting at 0 (Matlab convention starting at 1)';
eInfoDesc.value.LongName = 'event marker';
eInfoDesc.value.Description = 'marker value associated with the event';
eInfoDesc.value.Levels.x1   = 'MotorImageryA';
eInfoDesc.value.Levels.x2   = 'IdleStateA';
eInfoDesc.value.Levels.x3   = 'MotorImageryB';
eInfoDesc.value.Levels.x4   = 'IdleStateB';
eInfoDesc.value.Levels.x101   = 'MotorImageryA_detected';
eInfoDesc.value.Levels.x102   = 'IdleStateA_detected';
eInfoDesc.value.Levels.x103   = 'MotorImageryB_detected';
eInfoDesc.value.Levels.x104   = 'IdleStateB_detected';
eInfoDesc.value.Levels.x200   = 'Resting';
eInfoDesc.value.Levels.x201   = 'Fixation';
eInfoDesc.value.Levels.x202   = 'Action_observation';
eInfoDesc.value.Levels.x203   = 'Imagining';

% Content for README file
% -----------------------
README = sprintf( [ 'The experiments were conducted in an acoustically isolated room where only the participant and the experimenter were present. Participants voluntarily signed an informed consent form in accordance with the experimental protocol approved by the ethics committee of the Universidad Antonio Nariño. The participant was seated in a chair in a posture that was comfortable for him/her but did not affect data collection. In front of the participant, a 40-inch TV screen was placed at about 3 m. On this screen, a graphical user interface (GUI) displayed images that guided the participant through the experiment. Each experimental session was divided into two phases: an offline phase and an online phase.\n\n' ...
    'The offline experiments consisted of recording participants´ EEG signals during motor imagery trials for standing and sitting that were guided by the GUI presented on the TV screen. Six offline runs were conducted in which the participants were standing in three runs and sitting in the other three runs. In each run, the participant had to repeat a block of 30 trials of mental tasks indicated by visual cues continuously presented on the screen in a pseudo-random sequence.\n\n' ...
    'The first phase of the experimental session was conducted to construct the offline parts of the dataset: (A) Sit-to-stand and (B) Stand-to-sit. The participant´s EEG data were collected from 90 sequences for part A (45 trials of MotorImageryA tasks and 45 trials of IdleStateA tasks) and 90 sequences for part B (45 trials of MotorImageryB tasks and 45 trials of IdleStateB tasks).\n\n' ...
    'For each participant, the two machine learning models obtained in the offline phase were used to carry out the online experiment parts of the dataset: (C) Sit-to-stand and (D) Stand-to-sit. Each participant was instructed to select, in no particular order, 30 sequences for part C (15 trials of MotorImageryA tasks and 15 trials of IdleStateA tasks) and 30 other sequences for part D (15 trials of MotorImageryB tasks and 15 trials of IdleStateB tasks). Each trial was unique and was generated pseudo-randomly before the experiment.\n\n' ...
    'The database consisted of 32 electroencephalographic files corresponding to the 32 participants. All recordings were collected on channels F3, Fz, F4, FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6, P3, Pz, and P4 according to the 10-20 EEG electrode placement standard, grounded to AFz channel and referenced to right mastoid (M2). Each data file contained the data stream in a 2D matrix where rows corresponded to channels and columns corresponded to time samples with a sampling frequency of 250Hz.\n\n' ...
    'The following marker numbers encoded information about the execution of the experiment. Marker numbers 200, 201, 202, and 203, indicated the beginning and end of the four steps of the sequence in a trial (resting, fixation, action observation, and imagining). Marker numbers 1, 2, 3, and 4, indicated the figure activated on the screen to the participant perform the task corresponding to 1. actively imagining the sit-to-stand movement (labeled as MotorImageryA), 2. sitting motionless without imagining the sit-to-stand movement (labeled as IdleStateA), 3. standing motionless while actively imagining the stand-to-sit movement (labeled as MotorImageryB), or 4. standing motionless without imagining the stand-to-sit movement (labeled as IdleStateB). Finally, marker numbers 101, 102, 103, and 104, indicated the task detected by the BCI in real time during the online experiment: 101. MotorImageryA, 102. IdleStateA, 103. MotorImageryB, or 104. IdleStateB.' ] );

% Task information for xxxx-eeg.json file
% ---------------------------------------
tInfo.InstitutionAddress = 'Carrera 3 Este # 47 A - 15, Sede Circunvalar, Bogotá, Colombia';
tInfo.InstitutionName = 'Universidad Antonio Nariño';
tInfo.InstitutionalDepartmentName = 'Doctorado en Ciencia Aplicada';
tInfo.TaskName = 'sitstand';
tInfo.PowerLineFrequency = 60;
tInfo.SoftwareFilters.FilterDescription.Description = 'Bandpass filter using a 6th-order Butterworth filter with cutoff frequencies from 0.01 Hz to 60 Hz';
tInfo.ManufacturersModelName = 'g.Nautilus PRO';
tInfo.EEGPlacementScheme = '10-20';
tInfo.EEGReference = 'right mastoid (M2)';
tInfo.EEGGround = 'AFz channel';
tInfo.Manufacturer = 'g.tec';
tInfo.EEGChannelCount = 17;
tInfo.Instructions = 'First, a cross symbol was displayed on the TV screen and the participant was instructed to remain relaxed, motionless, and focused while looking at the symbol. Second, a figure appeared on the TV screen that the participant had to observe and then perform an experimental task. Third, the participant was instructed to act as the figure in order to perform the experimental task. For example, the figure could be sitting motionless while actively imagining the sit-to-stand movement, sitting motionless without imagining the sit-to-stand movement, standing motionless while actively imagining the stand-to-sit movement, or standing motionless without imagining the stand-to-sit movement. Finally, the text ´Descanso´ (the Spanish word for ´rest´) appeared on the TV screen, instructing the participant to rest from the experimental task, blink, or move the head and body as needed.';
tInfo.TaskDescription = sprintf( [ 'The temporal sequence task of a trial performed by each participant consisted of four steps. ' ...
    'As a first step (fixation), a cross symbol appeared on the TV screen for 4 s during which the participant was asked to avoid any body movement or effort and to stay focused while looking at the symbol. ' ...
    'In the second step (action observation), a figure appeared on the TV screen for 3 s, which the participant had to observe and perform one experimental task subsequently in the third step. ' ...
    'In the third step (imagining), the participant had to visualize the action indicated by the figure shown in the second step and perform one experimental task for 4 s in response to the figure. For instance, the task could be sitting motionless while actively imagining the sit-to-stand movement (labeled as MotorImageryA), sitting motionless without imagining the sit-to-stand movement (labeled as IdleStateA), standing motionless while actively imagining the stand-to-sit movement (labeled as MotorImageryB), or standing motionless without imagining the stand-to-sit movement (labeled as IdleStateB). ' ...
    'Finally, in the fourth step of the sequence (resting), the text ´Descanso´ (the Spanish word for ´Rest´) appeared on the TV screen for 4 s, instructing the participant to rest from the experimental task, blink, or move the head and body if necessary. ' ...
    'The difference between the offline and online timelines was in step 3. In the third step of the online timeline, the participant performed one experimental task in response to the figure shown in step 2, and the BCI attempted to detect this task in real time for 3-15 s at the same time that provided feedback.' ] );

% channel location file
% ---------------------
load(fullfile(sourcePath,'chanlocs','chanlocs.mat'));


% call to the export function
% ---------------------------
bids_export(data, ... 
    'targetdir', targetPath, ...
    'taskName', 'sitstand', ...
    'gInfo', generalInfo, ...
    'pInfo', pInfo, ...
    'pInfoDesc', pInfoDesc, ...
    'eInfoDesc', eInfoDesc, ...
    'README', README, ...
    'tInfo', tInfo, ...
    'chanlocs', chanlocs);

% remove stimuli folder and copy code file
% ----------------------
rmdir(char([targetPath,'stimuli']))
copyfile(fullfile(sourcePath, 'code', 'dataset_bids_export.m'), char([targetPath,'code']))


fprintf(2, 'WHAT TO DO NEXT?')
fprintf(2, ' -> upload the %s folder to http://openneuro.org to check it is valid\n', targetPath);
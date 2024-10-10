from .settings import Settings
from .utils import relative_to_root, fade_in, fade_out, leading_silence, trailing_silence
from .file_wrapper import AudioFile
from os import path
from pathlib import Path
from ruamel.yaml import YAML
import json
import pandas as pd
import copy

class ListeningTest(object):

  def __init__(self, settings: Settings) -> None:
    self.settings = settings

    audio_filename = Path(settings.get('filename')).stem
    self.run_name = f"SPLAT-{audio_filename}-{hash(self.settings)}"

    # listening_tests
    self.root_folder = relative_to_root("listening_tests")
    if not path.exists(self.root_folder):
      print("The folder %s does not exist", self.root_folder)
      return
    
    # listening_tests/db
    self.db_folder = self.root_folder.joinpath("db")
    self.db_file = self.db_folder.joinpath("webmushra.json")

    # listening_tests/webmushra
    self.webmushra_folder = self.root_folder.joinpath("webmushra")
    if not path.exists(self.webmushra_folder):
      print("Please install webmushra in the following directory: %s", self.webmushra_folder)
      return

    # listening_tests/webmushra/configs
    self.configs_folder = self.webmushra_folder.joinpath("configs")

    # listening_tests/webmushra/configs/resources
    self.resources_folder = self.configs_folder.joinpath("resources")

    # listening_tests/webmushra/configs/resources/audio
    self.audio_folder = self.resources_folder.joinpath("audio")

    # listening_tests/webmushra/configs/resources/stimuli
    self.stimuli_folder = self.resources_folder.joinpath("stimuli")

    # listening_tests/webmushra/configs/resources/stimuli/test_hash
    self.stimuli_test_folder = self.stimuli_folder.joinpath(self.run_name)
    self.stimuli_test_folder.mkdir(parents=True, exist_ok=True)

    # listening_tests/webmushra/configs/resources/references
    self.references_folder = self.resources_folder.joinpath("references")

    # listening_tests/webmushra/configs/resources/references/test_hash
    self.references_test_folder = self.references_folder.joinpath(self.run_name)
    self.references_test_folder.mkdir(parents=True, exist_ok=True)

  def _set_stimuli(self, stimuli_data, reference_file, reference=True) -> None:
    if reference:
      self.references = []
      destination = self.references
      test_folder = self.references_test_folder
    else:
      self.stimuli = []
      destination = self.stimuli
      test_folder = self.stimuli_test_folder
    
    fs = reference_file.get_samplerate()
    for idx, data in enumerate(stimuli_data):
      index, stimulus = data
      fade_time = 300
      fade_in(stimulus, fs, fade_time)
      fade_out(stimulus, fs, fade_time)
      # Add silence to the beginning and end of the stimulus
      stimulus = leading_silence(stimulus, fs, 200)
      stimulus = trailing_silence(stimulus, fs, 300)
      destination.append(AudioFile.from_audio_file(reference_file, new_data=stimulus, new_path=test_folder.joinpath(f"{idx}-{index}.wav")))

  def set_references(self, reference_data, reference_file) -> None:
    self._set_stimuli(reference_data, reference_file)

  def set_stimuli(self, stimuli_data, reference_file) -> None:
    self._set_stimuli(stimuli_data, reference_file, reference=False)

  def set_indexes(self, indexes) -> None:
    self.indexes = indexes

  def generate_config(self) -> None:

    # Generate the bulk of the config file
    config = {
      "testname": "Single Packet Loss Audibility Test",
      "testId": self.run_name,
      "bufferSize": 2048,
      "stopOnErrors": False,
      "showButtonPreviousPage": True,
      "remoteService": "service/write.php",
      "pages": []
    }

    iterations = self.settings.get("iterations")
    # Generate the first page with the examples
    explanation = {
      "type": "generic",
      "id": "Explanation",
      "name": "Instructions",
      "content": f'\
                <h2>Requirements</h2>\
                <p>For this test you will need a pair of Beyerdynamic DT770 PRO headphones, a good audio interface (avoid Behringer) and a quiet environment.</p>\
                <h2>Explanation</h2>\
                The goal of this test is to assess how audible audio glitches are in an audio signal.<br>\
                You will be presented with {self.settings.get("stimuli_per_page")+2} 3-second unlabeled audio tracks in each of the {self.settings.get("pages")*iterations} listening sessions.<br>\
                You need to assign each track a score between 0 and 100 representing how audible the glitch was.<br>\
                The scale is divided into the following five sections: "Not at all", "Barely", "A bit", "Much", and "Very Much".\
\
                  <h2>Audio Examples</h2>\
                  Below you can find some examples of the audio tracks you will be presented with.<br>\
                  Listen carefully to the examples to understand the scale.<br>\
                <table>\
                  <tr>\
                    <th></th>\
                    <th>Cello</th>\
                    <th>Piano</th>\
                  </tr>\
                  <tr>\
                    <td>Not at all</td>\
                    <td>\
                      <audio controls>\
                          <source src="configs/resources/audio/example_not_at_all_1_stim.wav" type="audio/wav">\
                          Your browser does not support the audio element.\
                      </audio>\
                    </td>\
                    <td>\
                      <audio controls>\
                          <source src="configs/resources/audio/example_not_at_all_2_ref.wav" type="audio/wav">\
                          Your browser does not support the audio element.\
                      </audio>\
                    </td>\
                  </tr>\
                  <tr>\
                    <td>A bit</td>\
                    <td>\
                      <audio controls>\
                          <source src="configs/resources/audio/example_a_bit_1_stim.wav" type="audio/wav">\
                          Your browser does not support the audio element.\
                      </audio>\
                    </td>\
                    <td>\
                      <audio controls>\
                          <source src="configs/resources/audio/example_a_bit_2_stim.wav" type="audio/wav">\
                          Your browser does not support the audio element.\
                      </audio>\
                    </td>\
                  </tr>\
                  <tr>\
                    <td>Very Much</td>\
                    <td>\
                      <audio controls>\
                          <source src="configs/resources/audio/example_very_much_1_stim.wav" type="audio/wav">\
                          Your browser does not support the audio element.\
                      </audio>\
                    </td>\
                    <td>\
                      <audio controls>\
                          <source src="configs/resources/audio/example_very_much_2_stim.wav" type="audio/wav">\
                          Your browser does not support the audio element.\
                      </audio>\
                    </td>\
                  </tr>\
                </table>\
                <style>\
                    table {{\
                        margin-left: auto;\
                        margin-right: auto;\
                        border-collapse: collapse;\
                    }}\
                    th, td {{\
                        border: 0px solid black;\
                        padding: 10px;\
                        text-align: center;\
                    }}\
                </style>'
    }
    config["pages"].append(explanation)

    consent = {
      "type": "generic",
      "id": "Consent",
      "name": "Declaration of consent",
      "content": "<h1>Declaration of consent</h1>\
      <p>By clicking the 'Next' button, you give your consent to the following. Please read the following consent entries carefully.</p>\
      <br/>\
      <p><i>I confirm that I have read and understand the participant information <br/>sheet for the project </i>Towards Perceptual Deep Learning-based Packet Loss Concealment.</p>\
      <br/>\
      <p><i>I confirm that I am 18 years of age or older.</i></p>\
      <br/>\
      <p><i>I understand that if I have any queries or concerns relating to this <br/>study I can contact Luca Vignati (luca.vignati@unitn.it) at any time.</i></p>\
      <br/>\
      <p><i>\
      I understand that I have the right to withdraw from this study at any <br/>\
      time without requirement to provide reason, and that if I withdraw,<br/> \
      any data provided up to this point will be retained.</i></p>\
      <br/>\
      <p><i>I agree for my personal information, including gender, age and listening experience to be recorded.</i></p>\
      <br/>\
      <p><i>\
      I understand that the data I provide may be presented anonymously in <br/>r\
      esearch output and publication, will be retained securely and indefinitely,<br/>\
      and may be made accessible to future researchers.</i></p>"
    }
    config["pages"].append(consent)

    volume = {
      "type": "generic",
      "id": "Volume",
      "name": "Adjust your volume",
      "content": "<h1>Adjust your volume</h1>\
      <p>Before the experiment begins, please adjust your listening volume.</p>\
      <p>Please do not change the output volume of the system from now on.</p>"
    }
    config["pages"].append(volume)

    # Generate the pages for the test
    reference = self.audio_folder.joinpath(self.settings.get("reference"))
    anchor = self.audio_folder.joinpath(self.settings.get("anchor"))
    randomized_pages = ["random"]
    page_content = "<p>IMPORTANT: When you press play on a stimulus, let it play till the end or hit pause. If you press play on another stimulus before the first finished, a glitch will be produced and the test will be invalid.</p>Please rate the audibility of the glitch in the following audio examples. How much can you hear it?"
    s = self.settings.get("stimuli_per_page")
    page_data = [(self.indexes[i:i+s], self.stimuli[i:i+s]) for i in range(0, len(self.references), s)]
    for data in page_data:
      indexes, stimuli = data
      page = {
        "type": "splat",
        "id": '-'.join([str(index) for index in indexes]),
        "name": "Test in progress",
        "content": page_content,
        "createAnchor35": False,
        "createAnchor70": False,
        "showWaveform": False,
        "enableLooping": False,
        "switchBack": True,
        "randomize": True,
        "reference": str(reference.relative_to(self.webmushra_folder)),
        "stimuli": {str(Path(stimulus.get_path()).stem): str(Path(stimulus.get_path()).relative_to(self.webmushra_folder)) for stimulus in stimuli}
      }
      page["stimuli"][str(anchor.stem)] = str(anchor.relative_to(self.webmushra_folder))
      randomized_pages.append(page)

    pause = {
      "type": "generic",
      "id": "Pause",
      "name": "Break",
      "content": "<h1>Break</h1>\
      <p>Please take 5 minutes of rest from this trial. Take off your headphones and chill for a while.</p>"
    }

    for _ in range(iterations):
      config["pages"].append(copy.deepcopy(randomized_pages))
      config["pages"].append(copy.deepcopy(pause))
    config["pages"].pop()

    # Generate the last page with the results"
    last_page = {
      "type": "finish",
      "name": "Thank you",
      "content": "Thank you for attending",
      "popupContent": "Your results were sent. Goodbye and have a nice day",
      "showResults": True,
      "writeResults": True,
      "questionnaire": [
        {
          "type": "text",
          "label": "Name",
          "name": "Name"
        },
        {
          "type": "text",
          "label": "Surname",
          "name": "Surname"
        },
        {
          "type": "number",
          "label": "Age",
          "name": "Age",
          "min": 0,
          "max": 100,
          "default": 30
        },
        {
          "type": "likert",
          "name": "gender",
          "label": "Gender",
          "response": [
            {
              "value": "male",
              "label": "Male"
            },
            {
              "value": "female",
              "label": "Female"
            }
          ]
        },
        {
          "type": "likert",
          "name": "Musical Proficiency",
          "label": "Musical Proficiency",
          "response": [
            {
              "value": "beginner",
              "label": "Beginner"
            },
            {
              "value": "intermediate",
              "label": "Intermediate"
            },
            {
              "value": "advanced",
              "label": "Advanced"
            }
          ]
        },
        {
          "type": "text",
          "label": "Email",
          "name": "email"
        }
      ]
    }
    config["pages"].append(last_page)
      
    yaml = YAML(typ=['rt', 'string'])
    config_file_path = self.configs_folder.joinpath(self.run_name + ".yaml")
    if config_file_path.exists():
      with open(config_file_path, 'r') as file:
        existing_config = yaml.load(file)
      if existing_config != config:
          raise ValueError(f"A config file with the same name and different content exists: {config_file_path}. Resolve manually.")
    else:
      with open(config_file_path, 'w') as file:
        config_stirng = yaml.dump_to_string(config).replace('- -', '-\n  -')
        file.write(config_stirng)

  def get_results(self) -> list:

    # Read the data from the database
    with open(self.db_file, 'r') as file:
      data = json.load(file)

    # Extract stimuli responses
    stimuli_responses = []
    for test_id, test_data in data.items():
        if self.run_name in test_id:
          for _, response_data in test_data.items():
              for response in response_data['responses']:
                    stimuli_responses.append({
                        'id': response['stimulus'],
                        'score': response['score']
                    })

    formatted_results = []
    if stimuli_responses:
      # Convert to DataFrame for easy grouping and calculation
      df = pd.DataFrame(stimuli_responses)

      df = df[~df['id'].str.contains('anchor|reference')]
      df = df.reset_index(drop=True)
      df['id'] = df['id'].str.split('-').str[-1]
      self.results = df

      # Group by id, calculate mean and std
      result = df.groupby('id').agg({
          'score': ['mean', 'std']
      }).reset_index()

      # Format as list of tuples
      formatted_results = list(result.itertuples(index=False, name=None))

    return formatted_results

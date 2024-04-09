# MOL3022 Project

## Installation
Install the required dependencies:
```
pip install -r requirements.txt
```

## Use
Run the program:
```
python .
```

Paste in a protein sequence (only the sequence itself, not a FASTA entry) in the input field, and press the 'Analyze sequence' button.  
In the 'Text output' window the sequence is printed out above the predicted structure, scroll horizontally if the sequence is wider than the window.
'C' is for coil, 'H' is for Helix, and 'B' is for Beta strand.

In the 'Graphical output' window, two graphs are presented.  
The top graph is a visual representation of the structure output. Coils are grey, Helixes are red, and Beta strands are blue.  
The bottom graph displays the models confidence in its prediction. The lighter the color is, the more confident the prediction is for that point.

![User interface](https://github.com/markuSolli/mol3022-project/blob/main/img/interface.png)

### Model training
A pre-trained model is included in the file `model_weights.pth`, and is used by the main script for prediction.  
If you want to train and analyze a model yourself, follow these steps.

#### Fetch data from UniProt
Create the `data` directory.  
Specify the number of pages to include in the training and validation data by editing the `TRAINING_N` and `TEST_N` fields in `.env` (1 page equals 500 protein entries).  
Finally:
```
python uniprot_parser.py
```

Verify that `training.csv` and `test.csv` are saved to the `data` directory.

#### (Optional) Adjust class weights
The current weights for each class are defined based on the response from UniProt when `TRAINING_N = 8` and `TEST_N = 2`.  
The distribution for other collections of data should be close, but may vary.  
Run:
```
python analyze_data.py
```
This outputs the appropiate weights for each class to the terminal.  
Enter the new values into the fields `WEIGHT_C`, `WEIGHT_H`, and `WEIGHT_B` in `.env`.

#### Train the model
To train the model and save the resulting weights to `model_weights.pth`, run:
```
python model.py -s
```
To validate the model against the test data, run:
```
python model.py -l
```

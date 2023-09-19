# Relating graph auto-encoders to linear models

## Code
All functionality from training is implemented (and well commented) in the gnn folder.


### run_batch.py 
- if you run it locally and for testing, you can specify parameters in the hard coded dictionary at the top of the file and run it as main
  - alternatively give as arguments via command line
  - refer parser.py function for default configurations of all parameters 
- to run several tasks on the server I created params files with all arguments for a single run in one line.
- output is automatically written to ./outputs containing the parameter configuration, training and testing results.  

### scripts
- calls the run_batch.py method on every parameter configuration given in textform
- all files contain absolute paths and need to be personalized

- Example for citeseer: 
```commandline
cd ..

while read p; do
	singularity exec --nv /YOUR_SINGULARITY_IMAGE.simg python3 PATH_TO/run_batch.py $p
	echo ONE MORE DONE!
done <~/PATH_TO_THE_PARAMETERS/experiments_citeseer.txt

echo ALL DONE!
```

### params
- .txt files containing one parameterconfiguration per line
- existing files contain experiments from paper


## Figures
- all figures from the paper can be automatically created via the figures.py 
- it will create .tex output but you can adapt the ending to generate .pdf or .png files

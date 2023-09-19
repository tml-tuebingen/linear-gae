#!/bin/bash

for f in *experiments*.sh; do
	sbatch "$f" &
done

echo "Ok now I'm ready!"

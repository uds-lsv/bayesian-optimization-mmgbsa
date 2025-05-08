1. Install [easydock](https://github.com/ci-lab-cz/easydock)
2. Extract the molecules from the `similarity_scores.csv` file (see screening/README.md)
    ```bash
    awk -F"," '{print $1}' screening/similarity_scores.csv > docking/molecules.smi
    ```
3. Split the file into chunks
    ```bash
    split molecules.smi -d --additional-suffix=.smi -n 5
    ```
4. Start docking
    ```bash
    ./start_docking
    ```
5. Once docking is finished run post-processing such as extracting the `.mol` blocks from the database. This produces the `results.csv` file.
    ```bash
    python extract_mols.py -db results/*.db
    ```


Note that the `molecules.smi` contains the protonated SMILES strings.
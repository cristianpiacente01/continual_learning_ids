# Master's thesis about CL for IDS

Repository containing a pipeline with experiments for my Master's thesis about Continual Learning in the Intrusion Detection Systems domain.

Cristian Piacente 866020 @ University of Milan - Bicocca

## How to Run

### Setup Virtual Environment

First of all, it's necessary to create a Python virtual environment and to install all the dependencies.

#### Creating and activating

Using the repository **root directory** as the working directory, execute the following command to **create the virtual environment**:

    python -m venv venv

Once the virtual environment has been created, it will be saved in the filesystem, so it won't be necessary to recreate it every time.

Now, it's possible to **activate it** with

    source ./venv/Scripts/activate

Please note: if you are on Windows, remove `source` from the command.

#### Installing the dependencies

After the **first activation**, it's necessary to **install all the needed libraries** by executing

    pip install -r requirements.txt

Once the dependencies have been installed, it won't be necessary anymore for the future activations.
With the virtual environment activated, it's possible to execute the different parts of the project, described in the other sections (TODO).

#### Deactivating

To quit the virtual environment, it's sufficient to execute the command

    deactivate

### Pipeline (TODO)

TODO

#### Execution

With the virtual environment activated in the repository root, to execute the Luigi pipeline run the following commands:

 1.     cd pipeline
 2.     python -m luigi --module pipeline Test --hello "testing the param value" --local-scheduler
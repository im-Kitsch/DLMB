# Skin-Lesion-Generator

## Project Structure
- Utility files (Dockerfile etc.)
- **rawdata** (contains images and meta data)
- **Generator** (contains the different code modules)
    - **dataloader** (module to abstract the sample loading)
    - **main.py** entry script
    
## Execution & Build
In order to be able to run the project one needs to install the requirements.
To have an easy-to-use setup one may build a docker image and run the python script within the docker runtime

### Docker build
```bash 
docker build -t skin-lesion-generator . 
```

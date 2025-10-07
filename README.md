# Ultimate RVC

[![PyPI version](https://badge.fury.io/py/ultimate-rvc.svg)](https://badge.fury.io/py/ultimate-rvc)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackismyShephard/ultimate-rvc/blob/main/notebooks/ultimate_rvc_colab.ipynb)
[![](https://dcbadge.limes.pink/api/server/https://discord.gg/T4ejEz8HtX?style=flat&compact=true&theme=default-inverted)](https://discord.gg/https://discord.gg/T4ejEz8HtX)
[![Open In Huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/JackismyShephard/ultimate-rvc)

An extension of [AiCoverGen](https://github.com/SociallyIneptWeeb/AICoverGen), which provides several new features and improvements, enabling users to generate audio-related content using RVC with ease. Ideal for people who want to incorporate singing functionality into their AI assistant/chatbot/vtuber, hear their favourite character sing their favourite song or have their favorite character read their favorite books aloud.

![ ](images/webui_generate.png?raw=true)

## Youtube Showcase

* [Ultimate RVC: The Most Powerful AI Voice Cloning Tool Yet!](https://www.youtube.com/watch?v=dVpCF68JpNg)
* [Ultimate-RVC on Your PC | Turn Any Voice Into Yours!](https://www.youtube.com/watch?v=IxFgz50y1AY)
* [Train Any Voice with Ultimate RVC | Fast & Accurate Voice Cloning](https://www.youtube.com/watch?v=O7VeYLL3H-c)

Courtesy of [Social & Apps](https://www.youtube.com/@socialapps1194)

## New Features

* Easy and automated setup using launcher scripts for both windows and Debian-based linux systems
* Significants improvements to voice conversion quality and speed. New features include support for additional pitch extraction methods such as FCPE, different embedder models and pre/post-processing options such as autotuning and noise reduction.
* TTS functionality, which allows you to generate speech from text using any RVC-based voice model. With this feature, you can do things such as generate audio books using your favourite character's voice.
* A voice model training suite, which allows you to train your own voice models using a wide range of options, such as different datasets, embedder models, and training configurations.
* Caching system which saves intermediate audio files as needed, thereby reducing inference time as much as possible. For example, if song A has already been converted using model B and now you want to convert song A using model C, then vocal extraction can be skipped and inference time reduced drastically
* Ability to listen to intermediate audio files in the UI. This is useful for getting an idea of what is happening in each step of a given generation pipeline.
* "multi-step" generation tabs: Here you can try out each step of a given generation pipeline in isolation. For example, if you already have extracted vocals available and only want to convert these using your voice model, then you can do that in a dedicated "multi-step" tab for song cover generation. Besides, these "multi-step" generation tabs are also useful for experimenting with settings for each step in a given generation pipeline.
* Lots of visual and performance improvements resulting from updating from Gradio 3 to Gradio 5 and from python 3.9 to python 3.12
* A redistributable package on PyPI, which allows you to easily access the Ultimate RVC project from any python 3.12 environment.
* Support for saving and loading of custom configurations for the Ultimate RVC web application. This allows you to easily switch between different configurations without having to manually change settings each time.

## Online Platforms

For those without a powerful enough NVIDIA GPU, you may try out Ultimate RV using [Google Colab](https://colab.research.google.com/github/JackismyShephard/ultimate-rvc/blob/main/notebooks/ultimate_rvc_colab.ipynb). Additionally, Ultimate RVC is also hosted on [Huggingface Spaces](https://huggingface.co/spaces/JackismyShephard/ultimate-rvc), although GPU acceleration is not available there. For those who want to run Ultimate RVC locally, follow the setup guide below.


## Local Setup

The Ultimate RVC project currently supports Windows and Debian-based Linux distributions, namely Ubuntu 22.04 and Ubuntu 24.04. Support for other platforms is not guaranteed.

To setup the project follow the steps below and execute the provided commands in an appropriate terminal. On windows this terminal should be **powershell**, while on Debian-based linux distributions it should be a **bash**-compliant shell.

### Install Git

Follow the [Git installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install Git on your computer.

### Set execution policy (Windows only)

To execute the subsequent commands on Windows, it is necessary to first grant
powershell permission to run scripts. This can be done at a user level as follows:

```console
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Clone Ultimate RVC repository

```console
git clone https://github.com/JackismyShephard/ultimate-rvc
cd ultimate-rvc
```

### Install dependencies

```console
./urvc install 
```

Note that on Linux, this command will install the CUDA 12.8 toolkit system-wide, if it is not already available. In case you have problems, you may need to install the toolkit manually.

### Start the app

```console
./urvc run
```

Once the output message `Running on local URL:  http://127.0.0.1:7860` appears, you can click on the link to open a tab with the web app.

### Update to latest version

```console
./urvc update
```

### Development mode

When developing new features or debugging, it is recommended to run the app in development mode. This enables hot reloading, which means that the app will automatically reload when changes are made to the code.

```console
./urvc dev
```

## Usage

### Manage models

#### Download models

![ ](images/webui_dl_model.png?raw=true)

Navigate to the `Download` subtab under the `Models` tab, and paste the download link to an RVC model and give it a unique name.
You may search the [AI Hub Discord](https://discord.gg/aihub) where already trained voice models are available for download.
The downloaded zip file should contain the .pth model file and optionally also a .index file.

Once the two input fields are filled in, simply click `Download`. Once the output message says `[NAME] Model successfully downloaded!`, you should be able to use the downloaded model in either the `Generate`>`song covers` or `Generate`>`speech` tab.

#### Upload models

![ ](images/webui_upload_model.png?raw=true)

For people who have trained RVC models locally and would like to use them for voice conversion.
Navigate to the `Upload` subtab under the `Models` tab, and follow the instructions there.
Once the output message says `Model with name [NAME] successfully uploaded!`, you should be able to use the uploaded model in either the `Generate`>`song covers` or `Generate`>`speech` tab.

### Generate song covers

#### One-click generation

![ ](images/webui_generate.png?raw=true)

* From the `Source type` dropdown, choose the source type from which you want to retrieve the song to convert.
* In the `Source` input field either paste the URL of a song on YouTube or upload an audio file, depending on the source type selected.
* From the `Voice model` dropdown menu, select the voice model to use.
* More Options can be viewed by clicking `Options`.

Once all options are filled in, click `Generate` and the AI generated song cover should appear in less than a few minutes, depending on your GPU.

## PyPI package

The Ultimate RVC project is also available as a [distributable package](https://pypi.org/project/ultimate-rvc/) on [PyPI](https://pypi.org/).

### Installation

The package can be installed with pip in a **Python 3.12**-based environment. To do so requires first installing PyTorch with CUDA support:

```console
pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

The Ultimate RVC project package can then be installed as follows:

```console
pip install ultimate-rvc
```

### CLI Usage

The `ultimate-rvc` package can be used as a python library but is primarily intended to be used as a command line tool. The package exposes two top-level commands:

* `urvc` which lets the user generate song covers directly from their terminal
* `urvc-web` which starts a local instance of the Ultimate RVC web application

For more information on either command supply the option `--help`.

## Environment Variables

The behaviour of the Ultimate RVC project can be customized via a number of environment variables. Currently these environment variables control only logging behaviour and data directory locations. They are as follows:

* `URVC_CONSOLE_LOG_LEVEL`: The log level for console logging. If not set, defaults to `ERROR`.
* `URVC_FILE_LOG_LEVEL`: The log level for file logging. If not set, defaults to `INFO`.
* `URVC_LOGS_DIR`: The directory in which log files will be stored. If not set, logs will be stored in a `logs` directory in the current working directory.
* `URVC_NO_LOGGING`: If set to `1`, logging will be disabled.
* `URVC_MODELS_DIR`: The directory in which models will be stored. If not set, models will be stored in a `models` directory in the current working directory.
* `URVC_AUDIO_DIR`: The directory in which audio files will be stored. If not set, audio files will be stored in an `audio` directory in the current working directory.
* `URVC_TEMP_DIR`: The directory in which temporary files will be stored. If not set, temporary files will be stored in a `temp` directory in the current working directory.
* `URVC_CONFIG_DIR`: The directory in which configuration files will be stored. If not set, configuration files will be stored in a `configs` directory in the current working directory.
* `URVC_VOICE_MODELS_DIR`: The directory in which voice models will be stored. If not set, voice models will be stored in a `voice_models` subdirectory of the `URVC_MODELS_DIR` directory.
* `YT_COOKIEFILE`: The path to a file containing cookies to use when downloading audio from YouTube via the web UI. If not set, no cookies will be used.
* `URVC_ACCELERATOR`: The type of hardware accelerator to use when running Ultimate RVC directly via the shell scripts in this repository. Currently supported options are `cuda` and `rocm`, with `cuda` being the default. Note that `rocm` is not supported on Windows and experimental on linux.
* `URVC_CONFIG`: The name of a configuration with custom values for settings to load when starting the Ultimate RVC web application. If not set, the default configuration for Ultimate RVC will be used. The configuration should be located in the `configs` directory of the Ultimate RVC project. If it does not exist, an error will be raised.

## Help & Community

If you experience (reproducible) bugs or have specific feature requests we encourage you to file a new issue here on GitHub. For broader discussions on the development of Ultimate RVC, as well as connecting with other users, we encourage you to join our Discord [server](https://discord.gg/T4ejEz8HtX) or check out the [Discussions](https://github.com/JackismyShephard/ultimate-rvc/discussions) page here on GitHub.

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.

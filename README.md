# VA-AM

<img src="https://raw.githubusercontent.com/cosminmarina/va_am/master/docs/_static/distribution.png" width="200" /> <img src="https://raw.githubusercontent.com/cosminmarina/va_am/master/docs/_static/identification.png" width="200" /> <img src="https://raw.githubusercontent.com/cosminmarina/va_am/master/docs/_static/identification2.png" width="200" /> <img src="https://raw.githubusercontent.com/cosminmarina/va_am/master/docs/_static/distribution2.png" width="200" />

Documentation[#](#documentation "Permalink to this heading")
--------------

The documentation is available [here](https://va-am.readthedocs.io/).

- [API Reference](https://va-am.readthedocs.io/en/latest/modules.html)
- [How to](https://va-am.readthedocs.io/en/latest/how_to.html)
- [License](https://va-am.readthedocs.io/en/latest/license.html)

Description[#](#description "Permalink to this heading")
------------

<img src="https://raw.githubusercontent.com/cosminmarina/va_am/master/docs/_static/va-am.png" width="200" />

VA-AM (Various Advanced - Analogue Methods) is a Python package based on the deep learning enhancement of the classical statistical **Analogue Method**. It provides several tools to analyse climatological **extreme events**, particularly **heat waves** (HW from now on).

It alows you to perform the identification of the HW following [Russo index](https://iopscience.iop.org/article/10.1088/1748-9326/10/12/124003), use the classical [Analogue Method](https://journals.ametsoc.org/view/journals/clim/12/8/1520-0442_1999_012_2474_tamaas_2.0.co_2.xml), use the enhanced Autoencoder Analogue Method, and even define own/use diferent deep learning architectures for the Analogue search.

Installation[#](#installation "Permalink to this heading")
-------------

Latest version:

Using pip

```
pip install va_am
```

Using conda

```
conda install -c conda-forge va_am
```

Latest commit:

```
pip install git+https://github.com/cosminmarina/va_am
```

Getting Started[#](#getting-started "Permalink to this heading")
----------------

VA-AM can be used inside a python code as library, or directly outside of the code, as a executable. See both options:

### Outside of code[#](#outside-of-code "Permalink to this heading")

A quick way of using it directly from your terminal. First try the -h | --help flag as:

```
python -m va_am -h
```

Note

You should obtain something like:

```
usage: __main__.py [-h] [-i] [-m METHOD] [-f CONF] [-sf SECRET] [-v] [-t]
           [-p PERIOD] [-sr]

optional arguments:
-h, --help            show this help message and exit
-i, --identifyhw      Flag. If true, first, identify the heatwave period
                        and, then, apply the 'method' if is one of: 'days',
                        'seasons', 'execs', 'latents', 'seasons-execs',
                        'latents-execs' or 'latents-seasons-execs'
-m METHOD, --method METHOD
                        Specify an method to execute between: 'day' (default),
                        'days', 'seasons', 'execs', 'latents', 'seasons-
                        execs', 'latents-execs' or 'latents-seasons-execs'
-f CONF, --configfile CONF
                        JSON file with configuration of parameters. If not
                        specified and 'method' require the file, it will be
                        searched at 'params.json'
-sf SECRET, --secretfile SECRET
                        Path to TXT file with needed information of the
                        Telegram bot to use to WARN and advice about
                        Exceptions. If not specified and 'method' require the
                        file, it will be searched at 'secret.txt'
-v, --verbose         Flag. If true, overwrite verbose param.
-t, --teleg           Flag. If true, exceptions and warnings will be sent to
                        Telegram Bot.
-p PERIOD, --period PERIOD
                        Specify the period where to perform the operations
                        between: 'both' (default), 'pre' or 'post'
-sr, --savereconstruction
                        Flag. If true, the reconstruction per iteration would
                        be saved in ./../../data/ folder as an
                        reconstruction-[name]-[day]-[period]-[AM/VA-AM].nc
                        file.
```

### Inside of code[#](#inside-of-code "Permalink to this heading")

You can import [va_am](https://va-am.readthedocs.io/en/latest/va_am.html) as a library in your code and use the equivalent method:

```
from va_am import

# Perform Variational Autoencoder Analogue search with default args
va_am()
```

or

```
import va_am

# Perform Variational Autoencoder Analogue search with default args
va_am.va_am()
```

Note

The arguments of `va_am()` method are the same as the outside of code version. For more details see the [API reference](https://va-am.readthedocs.io/en/latest/va_am.html).

Collaboration[#](#collaboration "Permalink to this heading")
--------------

If you find any bugs/issues or have any suggestions, please open an [issue](https://github.com/cosminmarina/va_am/issues/new).

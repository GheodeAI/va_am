How to...?
==========

.. contents::
    :local:

.. _config:

Configuration file
------------------
Regardless the way you will use :doc:`VA-AM <index>` (inside or outside a python code), you will need
a configuration file. It has to be a `JSON <https://en.wikipedia.org/wiki/JSON>`_ file with a structure as:

.. code-block:: json

    {
        "season":                   "all",
        "name":                     "-city_athens1987-",
        "latitude_min":             28,
        "latitude_max":             66,
        "longitude_min":            -8,
        "longitude_max":            50,
        "pre_init":                 "1851-01-06",
        "pre_end":                  "1950-12-31",
        "post_init":                "1951-01-01",
        "post_end":                 "2014-12-28",
        "data_of_interest_init":    "1987-06-01",
        "data_of_interest_end":     "1987-08-31",
        "load_AE":                  false,
        "load_AE_pre":              true,
        "file_AE_pre":              "./models/AE_pre.h5",
        "file_AE_post":             "./models/AE_post.h5",
        "latent_dim":               600,
        "use_VAE":                  true,
        "with_cpu":                 false,
        "n_epochs":                 5,
        "k":                        20,
        "iter":                     1000,
        "interest_region":          [38,38,24,24],
        "resolution":               2,
        "interest_region_type":     "coord",
        "per_what":                 "per_day",
        "remove_year":              false,
        "replace_choice":           true,
        "arch":                     5,
        "verbose":                  true,
        "target_dataset":             "~/path/to/data/data_dailyMax_t2m_1940-2022.nc",
        "pred_dataset":              "~/path/to/data/prmsl.nc",
        "ident_dataset":            "~/path/to/data_dailyMax_t2m_1940-2022.nc",
        "target_var_name":            "t2m_dailyMax",
        "p":                        2,
        "enhanced_distance":        true,
        "compile_params":           {
                                        "optimizer":"adamax",
                                        "loss":"mape",
                                        "metrics":["mae", "mse"]
                                    },
        "fit_params":               {
                                        "batch_size":64,
                                        "shuffle":true,
                                        "validation_split":0.15
                                    }
    }

By the flag ``-f`` | ``--configfile`` or the ``config_file`` parameter you can provide the Path
to your .json file containing the configuration parameters. If not provided, the program will search
for a default ``params.json`` file in the directory.

Then, we provide a list of all posible parameters, the type of parameter and a brief description of
each one:



====================  ===================  ========================================== 
Parameter             Type                 Description
====================  ===================  ========================================== 
season                str or list of str   String (or list of strings) that Specify 
                                           in wich season to perform the method,
                                           between: ``spring``, ``summer``, ``autumn``
                                           , ``winter``, ``spring-summer``,
                                           ``autumn-winter`` or ``all`` period.
name                  str                  Arbitrary name for identification of the
                                           execution/simulation and result file.
latitude/longitude    int                  The defined search region in terms of 
                                           minimal and maximal latitude and 
                                           longitude.
interest_region       list of int          Defined interest region where the
                                           reconstruction has to be maded, in terms
                                           of initial and end latitude and logitude.
                                           In should be a subregion of the defined
                                           search region. Otherwise it could be
                                           also the entire search region, but not
                                           bigger that it. See
                                           ``interest_region_type`` parameter for 
                                           more details.
interest_region_type  str                  Define if the ``interest_region`` list
                                           refers to list/array index positions
                                           (``idx`` option) or to spatial 
                                           coordinates (``coord`` option).
resolution            int                  Coordinates resolution of the dataset
                                           (Defaul value ``2``).
pre/post              str                  String with datetime of start (_init) and
                                           end (_end) of what we consider ``pre`` and
                                           ``post`` industrial data of our datasets.
                                           We can divide the datasets in 2 different 
                                           states to analyse, or use only one of them
                                           (e.g. post) to analyse all your datasets.
period                str                  String that indicates in wich period the 
                                           analysis will be performed. If could be
                                           ``both`` (default), only ``pre`` or only
                                           ``post``.
data_of_interest      str                  Same as previous, but for specify which is
                                           your interest datetime. (See 
                                           :ref:`Identify <identify>`)
load_AE               bool                 Flag that specify if the VA sould be 
                                           loaded from the ``file_AE``. If ``false``,
                                           the VA would be re-trained.
load_AE_pre           bool                 Same as previous flag, but only for VA in 
                                           ``pre`` epoch.
file_AE               str                  Path to where to save the trained models
                                           of VA for ``pre`` and ``post``. If
                                           ``load_AE`` is true, also represents from
                                           where the models will be loaded.
latent_dim            int                  Latent (or code) dimension to which the 
                                           predictor/driver should be reduced (or 
                                           codified).
use_VAE               bool                 Flag. If ``true`` and the ``arch`` is
                                           compatible, it will use a Variational 
                                           Autoencoder instead of a normal
                                           Autoencoder architecture.
with_cpu              bool                 Flag that indicate if the CPU or GPU
                                           version of tensorflow should be used, in
                                           case of having (or not) a GPU.
n_epochs              int                  Number of maximum epoch of training step.
n_execs               int                  If method is one of ``execs``,
                                           ``seasons-execs``, ``latents-execs`` or
                                           ``latents-seasons-execs``, it indicates
                                           the number of executions to perform with 
                                           the model (Defaul value ``5``).
k                     int                  How many analogue situation to select from
                                           the nearest ones. If ``k = 3`` the method
                                           will select the 3 nearest analogue
                                           situations. (Default value is ``20``).
iter                  int                  Number of random extraction to perform
                                           from the ``k`` nearest analogues, in 
                                           order to make a reconstruction of the 
                                           event.
per_what              str                  String to specify if the analysis should 
                                           be diary (``per_day``), weekly
                                           (``per_week``), monthly (``per_month``).
                                           Until now, this are the available option.
                                           In later versions yearly analysis will
                                           be avaiable.
remove_year           bool                 Flag that indicates if the year of the 
                                           interest period should be removed entirely
                                           or not. If false, only the period between
                                           ``data_of_interest_init`` and
                                           ``data_of_interest_end`` will be removed 
                                           from the dataset.
replace_choice        bool                 Flag that determines if the ``iter``
                                           random selection have to be perfomed with 
                                           (``true``) or without (``false``)
                                           replacement.
arch                  int                  Wich architecture of the available has to
                                           to be used. See
                                           `section <https://va-am.readthedocs.io/en/
                                           latest/va_am.utils.html#va_am.utils.AutoEn
                                           coders.AE_conv>`_
                                           for the available architectures.
verbose               bool                 If ``true``, several prints and warnings
                                           during the exectution will be showed. Also
                                           can be controled by ``-v`` | ``--verbose``
                                           flag or ``verbose`` parameter of the 
                                           outside and inside code execution of
                                           program.
target/pred_dataset   str                  Path to target (target) and predictor/driver
                                           (pred) datasets (``netcdf4`` or ``grib``).
ident_dataset         str                  Path to dataset where the identification
                                           will be performed. It could be the same 
                                           (or not) as the target dataset.
interest_dataset      str                  Path (optional) to the dataset of interest.
                                           That is, the one where occurs the event you
                                           are studing. Only specifie if you want to 
                                           extract the information od the interest
                                           from a different dataset than
                                           ``target_dataset``.
target_var_name       str                  Name of target variable in the dataset
                                           (default value if not specified is
                                           inferred from the dataset).
pred_var_name         str                  Name of predictor/driver variable in the 
                                           dataset. In case you don't specify it,
                                           the name will be inferred automatically.
                                           In future multi-variate VA-AM version,
                                           this parameter will change, probably to a
                                           list of strings or something like this.
interest_var_name     str                  Same case as ``target_var_name`` and 
                                           ``pred_var_name``, but only if
                                           ``interest_dataset`` is used.
p                     int                  Wich p-Minkowski distance to perform while
                                           the analog search, where taxicab
                                           distance is ``p=1``, euclidean distance is
                                           ``p=2``, and so on (default value ``2``)
enhanced_distance     bool                 Flag that indicates if the enhanced local
                                           proximity criterion should be used along
                                           with the p-Minkowski distance.
save_recons           bool                 Flag that indicates if the reconstruction
                                           of the target event should be saved
                                           (default value ``false``).
percentile            int                  Wich percentile should be used during the
                                           identification step (default value
                                           ``90``).
out_preprocess        str or list[str]     What to return from ``perform_preprocess``
                                           function. Default value is ``all``. The 
                                           possible output are: ``params``,
                                           ``img_size``, ``data_pred``, ``data_target``,
                                           ``time_pre_indust_pred``,
                                           ``time_indust_pred``,
                                           ``data_of_interest_pred``,
                                           ``data_of_interest_target``,
                                           ``x_train_pre_pred``, ``x_train_ind_pred``,
                                           ``x_test_pre_pred``, ``x_test_ind_pred``,
                                           ``pre_indust_pred``, ``pre_indust_target``,
                                           ``indust_pred``, ``indust_target``
compile_params        dict                 Dictionary wich contains the configuration
                                           input arguments for the
                                           `model.compile() <https://keras.io/api/mod
                                           els/model_training_apis/#compile-method>`_ 
                                           method, depending on the tensorflow/keras
                                           version.
fit_params            dict                 Dictionary wich contains the configuration
                                           input arguments for the `model.fit() <http
                                           s://keras.io/api/models/model_training_api
                                           s/#fit-method>`_ method, except for epochs 
                                           and verbose, depending on the
                                           tensorflow/keras version.
====================  ===================  ========================================== 


Functionality
-------------

This package provide, for now, the below functionality. More are expected in future versions.
The `github <https://github.com/cosminmarina/va_am>`_ repository have some example of
configuration files for some well known heat waves, but you should first check the
:ref:`Configuration file  <config>` section.

.. _identify:

Identify heat waves
*******************

We can perform the identifitacion of the heat wave period, following the definition from `Russo <http://doi.org/10.1088/1748-9326/10/12/124003>`_
paper. You will need a dataset of, ideally, maximum daily (or weekly) temperature as ``ident_dataset``.
From that you can perform the identification by by ``-i`` | ``--identifyhw`` flag or ``ident`` parameter as shown below,
with the corresponding :ref:`Configuration file  <config>`. 

.. code-block:: bash

    # Outside of the python code
    $ python -m va_am -i -f "path/to/config-file" ...

.. code-block:: python

    # Inside of the python code
    from va_am import va_am
    va_am(ident=True, config_file="path/to/config-file", ...)

Default methods of package are for :ref:`Analog search <analog-search>` or :ref:`Va-AM <va-am-methods>`,
so you can face 2 different scenarios: you will want to make de itentification as a first step of the 
other methods, or you will want to only make the identification.

In case you will use the identification as a first step of other methods, it is compatible with all methods
except ``day``. E.g., for method ``execs``:

.. code-block:: bash

    # Outside of the python code
    $ python -m va_am -i -m execs -f "path/to/config-file" ...

.. code-block:: python

    # Inside of the python code
    from va_am import va_am
    va_am(ident=True, method="execs", config_file="path/to/config-file",  ...)

In case you will use only the identification, is not required to specify any method. If the ``-i`` |
``--identifyhw`` flag is used, it will return a warning like ``Indentify Heat wave period (flag -i  
--identifyhw) for {params['name'][1:-1]} is not compatible with default 'method' ('day') and this
will not be executed`` indicating that only the identification is going to be performed (instead of
defauls ``day`` method).

.. code-block:: bash

    # Outside of the python code
    $ python -m va_am -i -f "path/to/config-file" ...

.. code-block:: python

    # Inside of the python code
    from va_am import va_am
    va_am(ident=True, config_file="path/to/config-file",  ...)

.. note::
    If Telegram bot is used you will also recive this warning. See :ref:`section <telegram>` for more details.

.. _analog-search:

Analog search
*************

The Analog method is a classic statistical search method based in a KNN search with a defined metric 
(See `Zorita <https://journals.ametsoc.org/view/journals/clim/12/8/1520-0442_1999_012_2474_tamaas_2.0.co_2.xml>`_
for a more detailed definition).

Until now, analog search is an auxiliar method that is not available from the outside python code versión.
It is expected that in next version of :doc:`VA-AM <index>`, the preprocess stage will be a more
generic one. With this, an only analog search method option will be allowed for outside python code
execution. For now, you can use it by:

.. code-block:: python

    from va_am import analogSearch
    analogSearch(...)

See the `corresponding API reference <https://va-am.readthedocs.io/en/latest/va_am.html#va_am.va_am.analogSearch>`_ for details about ``analogSearch`` arguments

.. _va-am-methods:

VA-AM methods
*************

The usual functionality of :doc:`VA-AM <index>` is to use `deep learning` methods (mainly Autoencoder-based)
to enhance the performance of the classic :ref:`analog <analog-search>`. We provide several already-done
architectures, such as `Variational-Autoencoder <https://va-am.readthedocs.io/en/latest/va_am.utils.html#va_
am.utils.AutoEncoders.AE_conv.kl_heatwave_arch_build>`_ , `Autoencoder <https://va-am.readthedocs.io/en/late
st/va_am.utils.html#va_am.utils.AutoEncoders.AE_conv.def_arch_build>`_, `Deep-Autoencoder <https://va-am.rea
dthedocs.io/en/latest/va_am.utils.html#va_am.utils.AutoEncoders.AE_conv.dense_build>`_, `Simetric-Autoencode
r <https://va-am.readthedocs.io/en/latest/va_am.utils.html#va_am.utils.AutoEncoders.AE_conv.batched_simetric
_build>`_, among others (see `API reference <https://va-am.readthedocs.io/en/latest/va_am.utils.html#va_am.u
tils.AutoEncoders.AE_conv>`_).

.. note::

    Where the order of architecture in the documentation correspond to its ``arch`` value in :ref:`Configuration file  <config>`.

For heat wave case a `specific architecture <https://va-am.readthedocs.io/en/latest/va_am.utils.html#va_am.u
tils.AutoEncoders.AE_conv.heatwave_arch_build>`_ is recommended (``arch=5``)

Is expected to implement in future versions a user-framework or method to use user-own architecture in :doc:`VA-AM <index>`.

.. _telegram:

Telegram bot
------------
:doc:`VA-AM <index>` include compatibility with a Telegram bot as warn and allert mechanism. It could
be useful when you are performing diferent long task and want to be notified about possibles
errors, exceptions and warnings.

To use it is quite easy by ``-t`` | ``--teleg`` flag or ``teleg`` parameter as shown below, but
first you will need to fulfill some previous steps:

.. code-block:: bash

    # Outside of the python code
    $ python -m va_am -t ...

.. code-block:: python

    # Inside of the python code
    from va_am import va_am
    va_am(..., teleg=True)


Step 1. Create your own Telegram bot
************************************
For the ``-t`` | ``--teleg`` option to work, you will need to create your own Telegram bot,
which will be who will notify you. *BotFather* is a built-in Telegram bot that allows you to
create another bots. We recommend to follow this `Tutorial <https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2>`_
in order to create the bot.

.. note::
    It is very important to save the **token** provided by *BotFather* of your Telegram bot.

Step 2. Create a channel or group
*********************************
The next step is to create a Telegram channel or group where you will get the allerts. We recommed
the use of a channel, but also a group could be possible. You will need to add your created bot
to this channel (or group) and allow it to send message (check the permissions you give to other
users/bots as admin of the channel).

When everything ready, you could follow the next step of the `Tutorial <https://medium.com/codex/using-python-to-send-telegram-messages-in-3-simple-steps-419a8b5e5e2>`_
to get the ``chat id``. Some snippet like the following could give you the ``chat id``:

.. code-block:: python

    import requests
    
    TOKEN = "YOUR TELEGRAM BOT TOKEN"
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    
    print(requests.get(url).json())

.. note::
    ``Chat id`` is an integer number that represents the channel (or group) which bot is member. It
    is important to Note that it could be a possitive or negative integer number, so be aware about
    the  ``-`` sign.

Step 3. Telegram secrets configuration file
*******************************************
The last step is to provide a secret file to the program to be able to use your Telegram bot.
By the flag ``-sf`` | ``--secretfile`` or the ``secret_file`` parameter you can provide the Path
to your .txt (or similar) file containing the secrets.

.. code-block:: bash

    # Outside of the python code
    $ python -m va_am -sr path/to/secret-file ...

.. code-block:: python

    # Inside of the python code
    from va_am import va_am
    va_am(..., secret_file="path/to/secret-file")

If not specified the secret file path, it will be searched at the default ``secret.txt`` file.

The scructure of the secret file need to be:

.. code-block:: none

    [TOKEN]
    [chat-id]
    @[user-name]


.. important::
    :doc:`VA-AM <index>` will send exceptions and warnings to the Telegram bot. In order to distinguish better
    exceptions from warnings, it use your ``[user-name]`` to notify you. If not wanted to follow this
    functionality, you could not provide it and replace ``@[user-name]`` by and empty space. 
    In any case, a third row is needed in the file, regardless it is empty, a white/blank space,
    or your ``@[user-name]``.

.. caution::
    **DON'T SHARE YOUR SECRET FILE WITH ANYONE!!!!**

    The ``[TOKEN]`` provides absolute access and admin permissions
    with your bot. In the wrong hands, it could end in a mess (probably your bot will became a spam bot,
    at best). If your going to use :doc:`VA-AM <index>` in a repository (especially a public one), we recommed you
    to add your secret file name to the `.gitignore <https://help.github.com/articles/ignoring-files>`_ file.


How to...?
==========

.. contents::
    :local:


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
        "temp_dataset":             "~/path/to/data/data_dailyMax_t2m_1940-2022.nc",
        "prs_dataset":              "~/path/to/data/prmsl.nc",
        "ident_dataset":            "~/path/to/data_dailyMax_t2m_1940-2022.nc",
        "temp_var_name":            "t2m_dailyMax",
        "p":                        2,
        "enhanced_distance":        true
    }

By the flag ``-f`` | ``--configfile`` or the ``config_file`` parameter you can provide the Path
to your .json file containing the configuration parameters. If not provided, the program will search
for a default ``params.json`` file in the directory.

Then, we provide a list of all posible parameters, the type of parameter and a brief description of
each one:


`str <https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str>`_
`list <https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range>`_
`int <https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex>`_
`float <https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex>`_
`bool <https://docs.python.org/3/library/stdtypes.html#boolean-type-bool>`_


==================  ===================  ========================================== 
Parameter           Type                 Description
==================  ===================  ========================================== 
season              str or list of str   String (or list of strings) that Specify 
                                         in wich season to perform the method,
                                         between: ``spring``, ``summer``, ``autumn``
                                         , ``winter``, ``spring-summer``,
                                         ``autumn-winter`` or ``all`` period.
name                str                  Arbitrary name for identification of the
                                         execution/simulation and result file.
latitude/longitude  int                  The defined search region in terms of 
                                         minimal and maximal latitude and 
                                         longitude.
==================  ===================  ========================================== 


Functionality
-------------

Identify heat waves
*******************

Analogue search
***************

VA-AM methods
*************

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


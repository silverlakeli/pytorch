import collections
import logging
import os
import re

DEFAULT_LOG_LEVEL = logging.WARN
DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s"
)

NAME_TO_LOG_NAME = {}
NAME_TO_RECORD_TYPE = {}
LOG_NAME_TO_REC_TYPES = collections.defaultdict(set)
# log names or artifact names can be part of the settings string
# dynamo + inductor logs have verbosity settings, aot only has one level
# names which support verbosity (prefix with a + lower or -)
# NB: this is the setting name
VERBOSE_NAMES = set()

# Set by user-facing API
settings = {}

# User API for setting log properties
# ex. format set_logs(LOG_NAME=LEVEL, ARTIFACT_NAME=bool)
# ex. set_logs(dynamo=logging.DEBUG, graph_code=True)
def set_logs(**kwargs):
    pass


# register a record type to be loggable
# setting name is the name used in the configuration env var
# log_name is the log that it belongs to
def loggable(setting_name, log_name, off_by_default=False):
    def register(cls):
        NAME_TO_LOG_NAME[setting_name] = log_name
        NAME_TO_RECORD_TYPE[setting_name] = cls

        # if off by default, don't enable it
        # when log_name's log_level is set to DEBUG
        if not off_by_default:
            LOG_NAME_TO_REC_TYPES[log_name].add(cls)

        return cls

    return register


def register_log(setting_name, log_name, has_levels=False):
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the name is associated with
        has_levels: whether the log supports different verbosity levels
    """
    NAME_TO_LOG_NAME[setting_name] = log_name
    if has_levels:
        VERBOSE_NAMES.add(setting_name)


def _get_loggable_names():
    return list(NAME_TO_LOG_NAME.keys()) + list(NAME_TO_RECORD_TYPE.keys())


VERBOSITY_CHAR = "+"
VERBOSITY_REGEX = re.escape(VERBOSITY_CHAR) + "?"

# match a comma separated list of loggable names (whitespace allowed after commas)
def gen_settings_regex(loggable_names):
    loggable_names_verbosity = [
        (VERBOSITY_REGEX if name in VERBOSE_NAMES else "") + name
        for name in loggable_names
    ]
    group = "(" + "|".join(loggable_names_verbosity) + ")"
    return re.compile(f"({group},\\s*)*{group}?")


def _validate_settings(settings):
    return re.fullmatch(gen_settings_regex(_get_loggable_names()), settings) is not None


def _parse_log_settings(settings):
    if settings == "":
        return dict()

    if not _validate_settings(settings):
        raise ValueError(
            f"Invalid log settings: {settings}, must be a comma separated list of registerered log or artifact names."
        )

    settings = re.sub(r"\s+", "", settings)
    log_names = settings.split(",")

    def get_name_level_pair(name):
        clean_name = name.replace(VERBOSITY_CHAR, "")
        level = None
        if clean_name in VERBOSE_NAMES:
            if name[0] == VERBOSITY_CHAR:
                level = logging.DEBUG
            else:
                level = logging.INFO

        return clean_name, level

    name_levels = [get_name_level_pair(name) for name in log_names]
    return {name: level for name, level in name_levels}


class FilterByType(logging.Filter):
    def __init__(self, enabled_types):
        self.enabled_types = tuple(set(enabled_types))

    def filter(self, record):
        return isinstance(record.msg, self.enabled_types)


def _get_log_settings():
    log_setting = os.environ.get("TORCH_LOGS", None)
    if log_setting is None:
        return {}
    else:
        return _parse_log_settings(log_setting)


# initialize loggers log_names
# each developer component should call this for their own logs
# in the appropriate location after relevant types have been registered
def init_logging(log_names, log_file_name=None, formatter=None):
    if not formatter:
        formatter = DEFAULT_FORMATTER

    name_to_levels = _get_log_settings()
    log_to_enabled_types = collections.defaultdict(set)
    log_name_to_level = dict()
    # only configure names associated with
    # log_names (ie, logs and artifacts associated with those log_names)
    allowed_names = set(
        log_names + [LOG_NAME_TO_REC_TYPES[log_name] for log_name in log_names]
    )

    # generate a map of log_name -> the types that should be logged
    for name, level in name_to_levels.items():
        if name not in allowed_names:
            continue

        if name not in NAME_TO_RECORD_TYPE:  # handle setting log settings
            log_name = NAME_TO_LOG_NAME[name]
            log_name_to_level[log_name] = level
            logging.getLogger(log_name).setLevel(
                logging.DEBUG
            )  # allow all messages through logger
            rec_types = []
            if level == logging.DEBUG:
                rec_types = LOG_NAME_TO_REC_TYPES[log_name]
                rec_types.add(str)
            log_to_enabled_types[log_name].update(rec_types)
        else:
            log_to_enabled_types[NAME_TO_LOG_NAME[name]].add(NAME_TO_RECORD_TYPE[name])

    # setup custom handlers
    # if the log level of a component is set to INFO, setup
    # an additional handler to print those messages, because
    # the debug handler is what handles custom objects like guards,
    # bytecode, etc.
    # if the log level of a component is set to DEBUG, allow all
    # string messages and allowed types (other than those off by default)
    def setup_handlers(create_handler_fn, log_name, enabled_types):
        log = logging.getLogger(log_name)
        debug_handler = create_handler_fn()
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        filter = FilterByType(enabled_types)
        debug_handler.addFilter(filter)
        log.addHandler(debug_handler)

        if (
            log_name in log_name_to_level
            and log_name_to_level[log_name] == logging.INFO
        ):
            info_handler = create_handler_fn()
            info_handler.setFormatter(formatter)
            info_handler.setLevel(logging.INFO)
            log.addHandler(info_handler)

    for log_name, enabled_types in log_to_enabled_types.items():
        setup_handlers(lambda: logging.StreamHandler(), log_name, enabled_types)

        if log_file_name is not None:
            setup_handlers(
                lambda: logging.FileHandler(log_file_name), log_name, enabled_types
            )

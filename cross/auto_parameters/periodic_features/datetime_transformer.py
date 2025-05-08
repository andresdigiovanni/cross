from cross.transformations import DateTimeTransformer
from cross.transformations.utils import dtypes
from cross.utils.verbose import VerboseLogger


class DateTimeTransformerParamCalculator:
    def calculate_best_params(
        self, x, y, model, scoring, direction, cv, groups, logger: VerboseLogger
    ):
        columns = dtypes.datetime_columns(x)

        logger.task_start("Detecting datetime features")
        total_columns = len(columns)

        if total_columns == 0:
            logger.warn("No datetime transformations was applied to any column")
            return None

        logger.task_result(
            f"Datetime transformations applied to {len(columns)} column(s)"
        )

        datetime_transformer = DateTimeTransformer(columns)
        return {
            "name": datetime_transformer.__class__.__name__,
            "params": datetime_transformer.get_params(),
        }

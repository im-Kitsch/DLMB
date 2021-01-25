import csv
import functools
import pandas
import numpy as np
from collections import namedtuple

LesionInfoTuple = namedtuple(
    'LesionInfoTuple',
    'lesion_id, image_id, disease_type, analytical_method, age, sex, localization',
)

PATH_TO_METADATA = "E:/datasets/HAM10000/HAM10000_metadata.csv"
#PATH_TO_METADATA = "/usr/src/rawdata/HAM10000_metadata.csv"

@functools.lru_cache(1)
def get_lesion_infos():
    df = pandas.read_csv(PATH_TO_METADATA)

    mean_age = df.groupby('sex', as_index=False).age.mean().age

    localizations = np.unique(df.localization)
    localizations = dict(zip(localizations, range(len(localizations))))

    dx = np.unique(df.dx)
    dx = dict(zip(dx, range(len(dx))))

    dx_type = np.unique(df['dx_type'])
    dx_type = dict(zip(dx_type, range(len(dx_type))))

    sex_type = {'female': 0, 'male': 1, 'unknown': 2}

    lesion_infos = []

    with open(PATH_TO_METADATA, "r") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row[0]) == 0:
                # handle empty rows
                continue

            lesion_id = row[0]
            image_id = row[1]
            disease = dx[row[2]]
            analytical_method = dx_type[row[3]]
            age = row[4]
            sex = sex_type[row[5]]

            if len(age) == 0:
                age = mean_age[sex]
            else:
                age = float(age)
                age = int(age)

            localization = localizations[row[6]]

            lesion_infos.append(
                LesionInfoTuple(lesion_id, image_id, disease, analytical_method, age, sex, localization)
            )
        return lesion_infos, dx, dx_type, sex_type, localizations


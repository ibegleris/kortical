import os
import pandas as pd
from kortical import api
from kortical.api.project import Project
from kortical.api.environment import Environment
from kortical.api.component_instance import ComponentInstanceState
from kortical.app import get_config
from alex.helpers.root_dir import from_root_dir

app_config = get_config(format='yaml')
data_file_name = app_config['data_file_name']
model_name = app_config['model_name']
target = app_config['target']

# To ensure the demo trains pretty quickly we:
#   - limit the solutions to linear models using TF-IDF and lemmatization
#   - Set random seeds
# Remove these restrictions for a full training run. To do this, simply remove the model code parameter, eg:
#   model = instance.train_model(
#       data,
#       number_of_train_workers=3,
#       target_score=0.97
#   )

model_code = """
- ml_solution:
  - data_set:
    - target_column: Category
    - problem_type: classification
    - evaluation_metric: f1_score
    - fraction_of_data_set_to_use: 1
    - cross_validation_folds: 3
    - data_set_random_seed: 1
    - modelling_random_seed: 1212414958
  - features:
    - numeric
    - categorical
    - text:
      - Text:
        - preprocessing:
          - custom:
            code: |
              @kortical.signature(input_columns=column_name, output_columns=[column_name, TEXT])
              def transform(column_name: str, input_columns: pd.DataFrame, fit_data):
                  lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
                  column = input_columns[column_name].map(lambda x: ' '.join([lmtzr.lemmatize(word) for word in x.split()]))
                  return {column_name: column}

            remove_original_column: False
        - create_features:
          - tf-idf
    - date:
  - models:
    - linear"""


if __name__ == '__main__':
    api.init()

    project = Project.get_selected_project()
    environment = Environment.get_selected_environment(project)

    # Load dataset
    df = pd.read_csv(from_root_dir(os.path.join("data", data_file_name)))

    # Create and save train/test split
    df_train = df.sample(frac=0.8)
    df_test = df.drop(df_train.index)
    df_train.to_csv(from_root_dir(os.path.join("data", data_file_name.replace('.csv', '_train.csv'))), index=False)
    df_test.to_csv(from_root_dir(os.path.join("data", data_file_name.replace('.csv', '_test.csv'))), index=False)

    # Do custom pre-processing (data cleaning / feature creation)

    # Create model
    instance = api.instance.Instance.create_or_select(model_name, delete_unpublished_models=True, stop_train=True)
    data = api.data.Data.upload_df(df_train, name=model_name)
    data.set_targets(target)

    model = instance.train_model(
        data,
        model_code=model_code,
        number_of_train_workers=3,
        target_score=0.965
    )

    print(f"Model score [{model.score} {model.evaluation_metric}]")

    component_instance = environment.create_component_instance(model.id)
    instance.set_default_version(model)
    component_instance.wait_for_status(ComponentInstanceState.RUNNING)

    print(f"Published model [{model.id} {model_name}] to environment [{environment}]")

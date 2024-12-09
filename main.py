import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import logging
from src.data_cleaning import DataFrameCleaner
from src.extra_features import Income_Municipality
from src.filtering import Postal_Filtering, One_Hot, BedroomsFiltering
from src.outliers import ZScoreFilter
from src.evaluation import ModelEvaluation
from src.visualizer import (
    LearningCurve,
    PredictionVsActualPlotter,
    FeatureImportanceVisualizer,
    ResidualsPlotter,
    PredictedPriceDistributionPlotter,
)


# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    """
    Class to handle data loading from file paths.
    """

    def __init__(self, dataset_path: str, income_mun_path: str) -> None:
        """
        Initializes the DataLoader with file paths for dataset and income data.

        :param dataset_path: Path to the main dataset CSV file.
        :param income_mun_path: Path to the income municipality CSV file.
        """
        self.dataset_path = dataset_path
        self.income_mun_path = income_mun_path

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads datasets from the provided file paths.

        :return: A tuple containing the main dataset and the income dataset as pandas DataFrames.
        """
        try:
            df = pd.read_csv(self.dataset_path)
            df_income = pd.read_csv(self.income_mun_path)
            return df, df_income
        except Exception as e:
            logging.error(f"Error loading data: {e}")


class DataPreProcessor:
    """
    Class to preprocess the data through a series of steps.
    """

    def __init__(self, df: pd.DataFrame, df_income: pd.DataFrame) -> None:
        """
        Initializes the DataPreProcessor with dataframes.

        :param df: The main dataset.
        :param df_income: The income dataset.
        """
        self.df = df
        self.df_income = df_income

    def preprocess(self) -> pd.DataFrame:
        """
        Applies all preprocessing steps to the dataset.

        :return: A cleaned and preprocessed DataFrame.
        """
        # Add extra feature based on income
        extra_feature = Income_Municipality(self.df, self.df_income)
        self.df = extra_feature.add_feature()

        # Filter postal codes
        postal_filter = Postal_Filtering(self.df, 10)
        self.df = postal_filter.postal_filtering()

        # Filter properties based on bedrooms
        bedrooms_filter = BedroomsFiltering(self.df, 8)
        self.df = bedrooms_filter.bedrooms_filtering()

        # Apply one-hot encoding to the 'province' column
        one_hot = One_Hot(self.df, "province")
        self.df = one_hot.one_hot_encoder()

        # Clean the dataset
        cleaner = DataFrameCleaner(self.df)
        self.df = cleaner.clean()

        # Remove outliers using Z-score
        zscore_filter = ZScoreFilter(
            self.df, columns=["price", "livingarea"], threshold=3
        )
        self.df = zscore_filter.filter()

        logging.info("Data preprocessing completed")
        return self.df


class ModelTrainer:
    """
    Class to train and evaluate machine learning models.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        """
        Initializes the ModelTrainer with training and test datasets.

        :param X_train: Features for training.
        :param X_test: Features for testing.
        :param y_train: Target variable for training.
        :param y_test: Target variable for testing.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(
        self, model: BaseEstimator, model_name: str
    ) -> BaseEstimator:
        """
        Trains and evaluates a model.

        :param model: A scikit-learn model.
        :param model_name: Name of the model for logging purposes.
        :return: The trained model.
        """
        # Train the model
        model.fit(self.X_train, self.y_train)

        # Evaluate the model
        evaluator = ModelEvaluation(
            model, self.X_train, self.X_test, self.y_train, self.y_test
        )
        print("Calling evaluator.print_metrics()...")
        evaluator.print_metrics()
        print("Finished calling evaluator.print_metrics()...")

        return model


class MainWorkflow:
    """
    Main workflow class to execute the entire pipeline.
    """

    def __init__(self, dataset_path: str, income_mun_path: str) -> None:
        """
        Initializes the workflow with file paths.

        :param dataset_path: Path to the main dataset.
        :param income_mun_path: Path to the income municipality dataset.
        """
        self.dataset_path = dataset_path
        self.income_mun_path = income_mun_path

    def execute(self) -> None:
        """
        Executes the workflow from data loading to model evaluation and visualization.
        """
        # Step 1: Load data
        loader = DataLoader(self.dataset_path, self.income_mun_path)
        df, df_income = loader.load()

        # Step 2: Preprocess data
        preprocessor = DataPreProcessor(df, df_income)
        df = preprocessor.preprocess()

        # Step 3: Split data
        X = df.drop(columns=["price"])
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Step 4: Train and evaluate models
        model_randomforest = RandomForestRegressor(random_state=42)
        model_decisiontree = DecisionTreeRegressor(random_state=42)

        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        logging.info("Training RandomForest model...")
        model_randomforest = trainer.train_and_evaluate(
            model_randomforest, "RandomForest"
        )

        logging.info("Training DecisionTree model...")
        model_decisiontree = trainer.train_and_evaluate(
            model_decisiontree, "DecisionTree"
        )

        # Step 5: Visualization
        logging.info("Visualizing")

        learningcurve_randomforest = LearningCurve(
            model_randomforest, X_train, y_train, X_test, y_test
        )
        learningcurve_randomforest.plot_learning_curve()
        learningcurve_decisiontree = LearningCurve(
            model_decisiontree, X_train, y_train, X_test, y_test
        )
        learningcurve_decisiontree.plot_learning_curve()

        featureofimportancevisualizer_randomforest = FeatureImportanceVisualizer(
            model_randomforest, X_train.columns
        )
        featureofimportancevisualizer_randomforest.plot()
        featureofimportancevisualizer_decisiontree = FeatureImportanceVisualizer(
            model_decisiontree, X_train.columns
        )
        featureofimportancevisualizer_decisiontree.plot()

        predictionvisualizer_randomforest = PredictionVsActualPlotter(
            model_randomforest, X_test, y_test
        )
        predictionvisualizer_randomforest.plot()
        predictionvisualizer_decisiontree = PredictionVsActualPlotter(
            model_decisiontree, X_test, y_test
        )
        predictionvisualizer_decisiontree.plot()

        residualsplotter_randomforest = ResidualsPlotter(
            model_randomforest, X_test, y_test
        )
        residualsplotter_randomforest.plot()
        residualsplotter_decisiontree = ResidualsPlotter(
            model_decisiontree, X_test, y_test
        )
        residualsplotter_decisiontree.plot()

        predicted_price_plotter_randomforest = PredictedPriceDistributionPlotter(
            model_randomforest, X_test
        )
        predicted_price_plotter_randomforest.plot()
        predicted_price_plotter_decisiontree = PredictedPriceDistributionPlotter(
            model_randomforest, X_test
        )
        predicted_price_plotter_decisiontree.plot()


if __name__ == "__main__":
    # File paths
    dataset_path = "./data/raw/dataset_province_municipality_code_large.csv"
    income_mun_path = "./data/raw/income_municipality.csv"

    # Execute the workflow
    workflow = MainWorkflow(dataset_path, income_mun_path)
    workflow.execute()

from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyDetector:
    """
    Detektor anomalii z flagowaniem obserwacji.
    Oznacza każdą próbkę w danych kolumną 'is_anomaly' = True/False.
    Umożliwia późniejsze sprawdzanie nowych punktów.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.is_fitted = False

    def fit_and_flag(self, data):
        """
        Trenuje model i dodaje kolumnę 'is_anomaly' do danych.
        
        Parametry:
        -----------
        data : pd.DataFrame lub np.ndarray
            Dane wejściowe (numeryczne cechy).
        
        Zwraca:
        --------
        flagged_data : pd.DataFrame
            Oryginalne dane + kolumna 'is_anomaly' (True jeśli anomalia).
        """
        # Konwersja do DataFrame jeśli trzeba
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Dopasowanie modelu
        self.model.fit(data)
        self.is_fitted = True
        
        # Predykcja (1 = normalny, -1 = anomalia)
        preds = self.model.predict(data)
        
        # Dodajemy kolumnę z flagą
        flagged_data = data.copy()
        flagged_data["flag"] = (preds == -1)
        
        return flagged_data

    def is_anomaly(self, new_point):
        """
        Sprawdza, czy nowy punkt jest anomalią (True/False).
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj fit_and_flag().")
        
        point_df = pd.DataFrame([new_point])
        pred = self.model.predict(point_df)[0]
        return pred == -1

    def anomaly_score(self, new_point):
        """
        Zwraca surowy wynik anomalii (im mniejszy, tym bardziej podejrzany punkt).
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj fit_and_flag().")
        
        point_df = pd.DataFrame([new_point])
        return self.model.decision_function(point_df)[0]

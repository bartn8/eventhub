import numpy as np
from typing import Dict, Optional
import random

class EventDropout:
    """
    Classe per applicare diverse strategie di dropout agli eventi.
    
    Gli eventi sono rappresentati come dizionario con chiavi 'x', 'y', 'p', 't'
    dove ogni valore è un array numpy di forma (N,) con N numero di eventi.
    """
    
    def __init__(self, height: int, width: int, dropout_p: float = 1.0, 
                 max_drop_count: int = 0, max_slice_size: int = 0, patch_size: int = 0, random_state: Optional[int] = None):
        """
        Inizializza l'EventDropout.
        
        Args:
            height: Altezza dell'immagine
            width: Larghezza dell'immagine  
            dropout_p: Probabilità di applicare il dropout (0.0 = mai, 1.0 = sempre)
            max_drop_count: Numero massimo di eventi da rimuovere nel random dropout
            max_slice_size: Dimensione massima della slice nel temporal dropout
            patch_size: Dimensione della patch nel spatial dropout
        """
        self.height = height
        self.width = width
        self.dropout_p = dropout_p
        self.max_drop_count = max_drop_count
        self.max_slice_size = max_slice_size
        self.patch_size = patch_size
        self.rng = np.random.default_rng(random_state)  # Usa Generator invece di RandomState

    def _should_apply_dropout(self) -> bool:
        """
        Determina se applicare il dropout in base alla probabilità dropout_p.
        """
        return self.rng.random() < self.dropout_p
    
    def random_dropout(self, events: Dict[str, np.ndarray], max_drop_count: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Rimuove un numero casuale di eventi.
        
        Args:
            events: Dizionario con eventi (chiavi: 'x', 'y', 'p', 't')
            max_drop_count: Numero massimo di eventi da rimuovere (se None, usa self.max_drop_count)
            
        Returns:
            Dizionario con eventi rimanenti dopo il dropout
        """
        # Controlla se applicare il dropout
        if not self._should_apply_dropout():
            return events
        
        if max_drop_count is None:
            max_drop_count = self.max_drop_count
        
        if max_drop_count <= 0:
            # Se non dobbiamo rimuovere nulla, restituisci gli eventi originali
            return events
        
        # Determina il numero effettivo di eventi da rimuovere (casuale fino a max_drop_count)
        drop_count = self.rng.integers(1, max_drop_count + 1)
        n_events = len(events['x'])
        if drop_count >= n_events:
            # Se dobbiamo rimuovere tutti gli eventi o più, restituisci eventi vuoti
            return {
                'x': np.array([], dtype=events['x'].dtype),
                'y': np.array([], dtype=events['y'].dtype), 
                'p': np.array([], dtype=events['p'].dtype),
                't': np.array([], dtype=events['t'].dtype)
            }
        # Seleziona indici casuali da rimuovere
        drop_indices = self.rng.choice(n_events, size=drop_count, replace=False)
        keep_mask = np.ones(n_events, dtype=bool)
        keep_mask[drop_indices] = False
        
        # Applica la maschera a tutti gli array
        filtered_events = {}
        for key in ['x', 'y', 'p', 't']:
            filtered_events[key] = events[key][keep_mask]
            
        return filtered_events
    
    def temporal_dropout(self, events: Dict[str, np.ndarray], max_slice_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Rimuove una slice temporale consecutiva di eventi.
        
        Args:
            events: Dizionario con eventi (chiavi: 'x', 'y', 'p', 't')
            max_slice_size: Dimensione massima della slice da rimuovere (se None, usa self.max_slice_size)
            
        Returns:
            Dizionario con eventi rimanenti dopo il dropout temporale
        """
        # Controlla se applicare il dropout
        if not self._should_apply_dropout():
            return events
        
        if max_slice_size is None:
            max_slice_size = self.max_slice_size
            
        n_events = len(events['x'])
        
        if max_slice_size <= 0:
            # Se non dobbiamo rimuovere nulla, restituisci gli eventi originali
            return events
        
        # Determina la dimensione effettiva della slice (casuale fino a max_slice_size)
        slice_size = self.rng.integers(1, max(max_slice_size, n_events) + 1)
        # Gli eventi dovrebbero essere ordinati per timestamp, quindi prendiamo una slice consecutiva
        # start_idx = self.rng.integers(0, n_events - slice_size + 1)
        start_idx = 0
        end_idx = start_idx + slice_size
        
        # Crea maschera per mantenere gli eventi fuori dalla slice
        keep_mask = np.ones(n_events, dtype=bool)
        keep_mask[start_idx:end_idx] = False
        
        # Applica la maschera a tutti gli array
        filtered_events = {}
        for key in ['x', 'y', 'p', 't']:
            filtered_events[key] = events[key][keep_mask]
            
        return filtered_events
    
    def spatial_dropout(self, events: Dict[str, np.ndarray], patch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Rimuove eventi in una patch spaziale casuale.
        
        Args:
            events: Dizionario con eventi (chiavi: 'x', 'y', 'p', 't')
            patch_size: Dimensione della patch quadrata (se None, usa self.patch_size)
            
        Returns:
            Dizionario con eventi rimanenti dopo il dropout spaziale
        """
        # Controlla se applicare il dropout
        if not self._should_apply_dropout():
            return events
        
        if patch_size is None:
            patch_size = self.patch_size
            
        if patch_size <= 0:
            return events
        
        # Seleziona centro della patch casualmente
        center_x = self.rng.integers(0, self.width)
        center_y = self.rng.integers(0, self.height)
        
        # Calcola i limiti della patch
        half_patch = patch_size // 2
        min_x = max(0, center_x - half_patch)
        max_x = min(self.width - 1, center_x + half_patch)
        min_y = max(0, center_y - half_patch) 
        max_y = min(self.height - 1, center_y + half_patch)
        
        # Crea maschera per mantenere eventi fuori dalla patch
        x_in_patch = (events['x'] >= min_x) & (events['x'] <= max_x)
        y_in_patch = (events['y'] >= min_y) & (events['y'] <= max_y)
        in_patch_mask = x_in_patch & y_in_patch
        keep_mask = ~in_patch_mask
        
        # Applica la maschera a tutti gli array
        filtered_events = {}
        for key in ['x', 'y', 'p', 't']:
            filtered_events[key] = events[key][keep_mask]
            
        return filtered_events
    
    def apply_random_dropout(self, events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # dropout_type = self.rng.choice(['random', 'temporal', 'spatial'])
        
        if self.rng.random() < 0.33:
            dropout_type = 'random'
        else:
            dropout_type = 'temporal'
        
        return self.apply_dropout(events, dropout_type)

    def apply_dropout(self, events: Dict[str, np.ndarray], 
                     dropout_type: str = 'random') -> Dict[str, np.ndarray]:
        """
        Applica il dropout specificato agli eventi usando i parametri del costruttore.
        
        Args:
            events: Dizionario con eventi (chiavi: 'x', 'y', 'p', 't')
            dropout_type: Tipo di dropout ('random', 'temporal', 'spatial')
        
        Returns:
            Dizionario con eventi dopo il dropout
        """
        if dropout_type == 'random':
            return self.random_dropout(events)
        elif dropout_type == 'temporal':
            return self.temporal_dropout(events)
        elif dropout_type == 'spatial':
            return self.spatial_dropout(events)
        else:
            raise ValueError(f"Tipo di dropout non supportato: {dropout_type}")


class RandomEventDropout(EventDropout):
    """Dropout casuale di eventi."""
    
    def __init__(self, height: int, width: int, max_drop_count: int, dropout_p: float = 1.0):
        super().__init__(height, width, dropout_p, max_drop_count=max_drop_count)
    
    def __call__(self, events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.random_dropout(events)


class TemporalEventDropout(EventDropout):
    """Dropout temporale di eventi."""
    
    def __init__(self, height: int, width: int, max_slice_size: int, dropout_p: float = 1.0):
        super().__init__(height, width, dropout_p, max_slice_size=max_slice_size)
    
    def __call__(self, events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.temporal_dropout(events)


class SpatialEventDropout(EventDropout):
    """Dropout spaziale di eventi."""
    
    def __init__(self, height: int, width: int, patch_size: int = 3, dropout_p: float = 1.0):
        super().__init__(height, width, dropout_p, patch_size=patch_size)
    
    def __call__(self, events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.spatial_dropout(events)


# Esempio di utilizzo
if __name__ == "__main__":
    # Crea eventi di esempio
    n_events = 1000
    rng = np.random.default_rng(42)
    events = {
        'x': rng.integers(0, 640, n_events),
        'y': rng.integers(0, 480, n_events), 
        'p': rng.choice([0, 1], n_events),
        't': np.sort(rng.integers(0, 1000000, n_events))
    }
    
    # Inizializza dropout con parametri nel costruttore
    dropout = EventDropout(height=480, width=640, dropout_p=0.5, 
                          max_drop_count=100, max_slice_size=50, patch_size=99)
    
    # Test random dropout
    print(f"Eventi originali: {len(events['x'])}")
    
    # Testa più volte per vedere l'effetto della probabilità e della casualità
    print("\nTest con dropout_p=0.5 e max_drop_count=100 (rimuove da 1 a 100 eventi casuali):")
    for i in range(5):
        random_events = dropout.random_dropout(events)
        print(f"Test {i+1} - Dopo random dropout: {len(random_events['x'])} eventi")
    
    # Test con dropout_p=1.0 (sempre)
    dropout_always = EventDropout(height=480, width=640, dropout_p=1.0, max_drop_count=100)
    print(f"\nCon dropout_p=1.0 e max_drop_count=100:")
    for i in range(3):
        random_events = dropout_always.random_dropout(events)
        events_removed = len(events['x']) - len(random_events['x'])
        print(f"Test {i+1} - Rimossi {events_removed} eventi, rimangono {len(random_events['x'])}")
    
    # Test con dropout_p=0.0 (mai)
    dropout_never = EventDropout(height=480, width=640, dropout_p=0.0, max_drop_count=100)
    random_events = dropout_never.random_dropout(events)
    print(f"\nCon dropout_p=0.0 - Dopo random dropout: {len(random_events['x'])} eventi")
    
    # Test usando apply_dropout
    temporal_events = dropout.apply_dropout(events, 'temporal')
    print(f"\nDopo temporal dropout (max_slice_size=50): {len(temporal_events['x'])}")
    
    spatial_events = dropout.apply_dropout(events, 'spatial')
    print(f"Dopo spatial dropout (patch_size=99): {len(spatial_events['x'])}")
    
    # Test delle classi specializzate
    print("\n--- Test classi specializzate ---")
    random_dropper = RandomEventDropout(height=480, width=640, max_drop_count=50, dropout_p=1.0)
    for i in range(3):
        result = random_dropper(events)
        events_removed = len(events['x']) - len(result['x'])
        print(f"RandomEventDropout test {i+1}: rimossi {events_removed} eventi, rimangono {len(result['x'])}")
    
    temporal_dropper = TemporalEventDropout(height=480, width=640, max_slice_size=30, dropout_p=1.0)
    result = temporal_dropper(events)
    print(f"\nTemporalEventDropout: {len(result['x'])} eventi")
    
    spatial_dropper = SpatialEventDropout(height=480, width=640, patch_size=20, dropout_p=1.0)
    result = spatial_dropper(events)
    print(f"SpatialEventDropout: {len(result['x'])} eventi")

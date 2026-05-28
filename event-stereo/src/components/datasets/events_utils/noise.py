import numpy as np
from typing import Dict, Optional

class EventNoise:
    """
    Classe per applicare rumore agli eventi spostandoli in posizioni casuali.
    
    Gli eventi sono rappresentati come dizionario con chiavi 'x', 'y', 'p', 't'
    dove ogni valore è un array numpy di forma (N,) con N numero di eventi.
    """
    
    def __init__(self, height: int, width: int, noise_p: float = 1.0, 
                 max_noise_count: int = 0, random_state: Optional[int] = None):
        """
        Inizializza l'EventNoise.
        
        Args:
            height: Altezza dell'immagine
            width: Larghezza dell'immagine  
            noise_p: Probabilità di applicare il rumore (0.0 = mai, 1.0 = sempre)
            max_noise_count: Numero massimo di eventi da spostare casualmente
            random_state: Seed per il generatore di numeri casuali
        """
        self.height = height
        self.width = width
        self.noise_p = noise_p
        self.max_noise_count = max_noise_count
        self.rng = np.random.default_rng(random_state)

    def _should_apply_noise(self) -> bool:
        """
        Determina se applicare il rumore in base alla probabilità noise_p.
        """
        return self.rng.random() < self.noise_p
    
    def add_spatial_noise(self, events: Dict[str, np.ndarray], max_noise_count: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Sposta un numero casuale di eventi in posizioni casuali.
        
        Args:
            events: Dizionario con eventi (chiavi: 'x', 'y', 'p', 't')
            max_noise_count: Numero massimo di eventi da spostare (se None, usa self.max_noise_count)
            
        Returns:
            Dizionario con eventi modificati dopo l'applicazione del rumore
        """
        # Controlla se applicare il rumore
        if not self._should_apply_noise():
            return events
        
        if max_noise_count is None:
            max_noise_count = self.max_noise_count
        
        if max_noise_count <= 0:
            # Se non dobbiamo spostare nulla, restituisci gli eventi originali
            return events
        
        noisy_events = events
        
        n_events = len(events['x'])
        if n_events == 0:
            return noisy_events
        
        # Determina il numero effettivo di eventi da spostare (casuale fino a max_noise_count)
        noise_count = self.rng.integers(1, min(max_noise_count + 1, n_events + 1))
        
        p = np.exp(-np.arange(n_events) / n_events)  # Probabilità decrescente
        p /= p.sum()  # Normalizza
        
        # Seleziona indici casuali da spostare
        noise_indices = self.rng.choice(n_events, size=noise_count, replace=False, p=p)
        
        # Genera nuove posizioni casuali per gli eventi selezionati
        new_x = self.rng.integers(0, self.width, size=noise_count)
        new_y = self.rng.integers(0, self.height, size=noise_count)
        
        # Applica il rumore alle posizioni
        noisy_events['x'][noise_indices] = new_x
        noisy_events['y'][noise_indices] = new_y
        # Manteniamo 'p' e 't' invariati
            
        return noisy_events
    
    def __call__(self, events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Applica il rumore spaziale agli eventi.
        
        Args:
            events: Dizionario con eventi (chiavi: 'x', 'y', 'p', 't')
        
        Returns:
            Dizionario con eventi dopo l'applicazione del rumore
        """
        return self.add_spatial_noise(events)


# Esempio di utilizzo
if __name__ == "__main__":
    # Crea eventi di esempio
    n_events = 100
    rng = np.random.default_rng(42)
    events = {
        'x': rng.integers(0, 640, n_events),
        'y': rng.integers(0, 480, n_events), 
        'p': rng.choice([0, 1], n_events),
        't': np.sort(rng.integers(0, 1000000, n_events))
    }
    
    print("Eventi originali (primi 10):")
    for i in range(10):
        print(f"Evento {i}: x={events['x'][i]}, y={events['y'][i]}, p={events['p'][i]}, t={events['t'][i]}")
    
    # Inizializza noise con parametri nel costruttore
    noise = EventNoise(height=480, width=640, noise_p=1.0, max_noise_count=20)
    
    # Test spatial noise
    print(f"\nEventi originali: {len(events['x'])}")
    
    # Testa più volte per vedere l'effetto della casualità
    print(f"\nTest con noise_p=1.0 e max_noise_count=20 (sposta da 1 a 20 eventi):")
    for i in range(3):
        noisy_events = noise.add_spatial_noise(events)
        
        # Conta quanti eventi sono stati effettivamente spostati
        moved_count = 0
        for j in range(len(events['x'])):
            if events['x'][j] != noisy_events['x'][j] or events['y'][j] != noisy_events['y'][j]:
                moved_count += 1
        
        print(f"Test {i+1} - Eventi spostati: {moved_count}")
        
        # Mostra alcuni esempi di eventi spostati
        print("  Esempi di eventi spostati (primi 5):")
        shown = 0
        for j in range(len(events['x'])):
            if events['x'][j] != noisy_events['x'][j] or events['y'][j] != noisy_events['y'][j]:
                print(f"    Evento {j}: ({events['x'][j]},{events['y'][j]}) -> ({noisy_events['x'][j]},{noisy_events['y'][j]})")
                shown += 1
                if shown >= 5:
                    break
    
    # Test con noise_p=0.5 (50% probabilità)
    noise_prob = EventNoise(height=480, width=640, noise_p=0.5, max_noise_count=10)
    print(f"\nTest con noise_p=0.5 e max_noise_count=10:")
    for i in range(5):
        noisy_events = noise_prob.add_spatial_noise(events)
        moved_count = sum(1 for j in range(len(events['x'])) 
                         if events['x'][j] != noisy_events['x'][j] or events['y'][j] != noisy_events['y'][j])
        print(f"Test {i+1} - Eventi spostati: {moved_count}")
    
    # Test con noise_p=0.0 (mai)
    noise_never = EventNoise(height=480, width=640, noise_p=0.0, max_noise_count=10)
    noisy_events = noise_never.add_spatial_noise(events)
    moved_count = sum(1 for j in range(len(events['x'])) 
                     if events['x'][j] != noisy_events['x'][j] or events['y'][j] != noisy_events['y'][j])
    print(f"\nCon noise_p=0.0 - Eventi spostati: {moved_count}")
    
    # Test usando __call__
    print(f"\nTest usando __call__:")
    noisy_events = noise(events)
    moved_count = sum(1 for j in range(len(events['x'])) 
                     if events['x'][j] != noisy_events['x'][j] or events['y'][j] != noisy_events['y'][j])
    print(f"Eventi spostati: {moved_count}")
    
"""
Étudiant : Azizi Oussama
Formation : M2 Mathématiques et Data Science
Introduction and Learnability - Solution
Université de Haute-Alsace
Statistical and Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt

# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)

def generate_data(m, theta=0.5, noise_rate=0.0):
    """
    Génère un jeu de données de taille m avec un classifieur seuil theta.
    Dans le cadre agnostique, les labels sont inversés indépendamment avec une probabilité noise_rate.
    """
    X = np.random.uniform(0, 1, size=m)
    y_true = (X > theta).astype(int)
    
    if noise_rate > 0.0:
        flip_mask = np.random.rand(m) < noise_rate
        y = np.where(flip_mask, 1 - y_true, y_true)
    else:
        y = y_true
        
    return X, y


class ThresholdERM:
    """
    Algorithme de Minimisation du Risque Empirique (ERM) pour les classifieurs seuils.
    """
    def __init__(self):
        self.threshold = None
        
    def fit(self, X, y):
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        
        # Trouver les points médians comme candidats pour le seuil
        if len(X) > 1:
            midpoints = (X_sorted[:-1] + X_sorted[1:]) / 2.0
            candidates = np.concatenate([[-0.1], midpoints, [1.1]])
        else:
            candidates = np.array([X[0] - 0.1, X[0] + 0.1])
            
        best_error = float('inf')
        best_c = candidates[0]
        
        # Évaluer chaque seuil candidat et retenir celui qui minimise l'erreur empirique
        for c in candidates:
            preds = (X > c).astype(int)
            error = np.mean(preds != y)
            if error < best_error:
                best_error = error
                best_c = c
                
        self.threshold = best_c
        return self.threshold

    def predict(self, X):
        return (X > self.threshold).astype(int)


def true_error_threshold(c, theta):
    """
    Calcule l'erreur théorique (vraie erreur) L(h) pour un seuil c par rapport au vrai seuil theta.
    Puisque X est uniformément réparti sur [0, 1], l'erreur est la longueur de l'intervalle symétrique entre c et theta.
    """
    return abs(c - theta)


def run_pac_experiments(theta=0.5, sample_sizes=[5, 10, 20, 50, 100, 200], 
                        num_repetitions=500, epsilon=0.1, noise_rate=0.0):
    """
    Exécute les expériences PAC / Agnostic PAC sur plusieurs tailles d'échantillons et répétitions.
    """
    results = {}
    
    print(f"\n--- Exécution des expériences (Bruit : {noise_rate}, Theta : {theta}) ---")
    for m in sample_sizes:
        exceed_epsilon_count = 0
        empirical_errors = []
        true_errors = []
        
        for _ in range(num_repetitions):
            X_train, y_train = generate_data(m, theta=theta, noise_rate=noise_rate)
            
            erm = ThresholdERM()
            erm.fit(X_train, y_train)
            
            train_preds = erm.predict(X_train)
            emp_err = np.mean(train_preds != y_train)
            empirical_errors.append(emp_err)
            
            true_err = true_error_threshold(erm.threshold, theta)
            true_errors.append(true_err)
            
            if true_err > epsilon:
                exceed_epsilon_count += 1
                
        prob_exceed = exceed_epsilon_count / num_repetitions
        results[m] = {
            'prob_exceed': prob_exceed,
            'mean_emp_error': np.mean(empirical_errors),
            'mean_true_error': np.mean(true_errors)
        }
        print(f"m = {m:3d} | P(L(h) > {epsilon}) = {prob_exceed:.3f} | Erreur Vraie Moyenne = {np.mean(true_errors):.3f} | Erreur Empirique Moyenne = {np.mean(empirical_errors):.3f}")
        
    return results


if __name__ == "__main__":
    theta = 0.5
    sample_sizes = [5, 10, 20, 50, 100, 200]
    epsilon = 0.1

    # --- Tâche 1 : Visualisation d'un jeu de données ---
    X_sample, y_sample = generate_data(50, theta=theta, noise_rate=0.0)
    
    plt.figure(figsize=(10, 4))
    plt.scatter(X_sample, y_sample, c=y_sample, cmap='bwr', edgecolor='k', s=50)
    plt.axvline(x=theta, color='green', linestyle='--', label=f'Vrai seuil theta = {theta}')
    plt.xlabel('X')
    plt.ylabel('Label y')
    plt.title('Visualisation d\'un jeu de données généré (m = 50, Réalisable)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Tâche 4 : Vérification empirique de l'apprentissage PAC (Cadre Réalisable) ---
    pac_results = run_pac_experiments(theta=theta, sample_sizes=sample_sizes, 
                                      num_repetitions=500, epsilon=epsilon, noise_rate=0.0)

    # --- Tâche 5 : Passage au cadre Agnostic PAC (Avec bruit de labels) ---
    agnostic_results = run_pac_experiments(theta=theta, sample_sizes=sample_sizes, 
                                         num_repetitions=500, epsilon=epsilon, noise_rate=0.1)

    # --- Visualisation comparative des probabilités d'erreur ---
    m_list = sample_sizes
    pac_probs = [pac_results[m]['prob_exceed'] for m in m_list]
    agnostic_probs = [agnostic_results[m]['prob_exceed'] for m in m_list]

    plt.figure(figsize=(8, 5))
    plt.plot(m_list, pac_probs, marker='o', label='Cadre Réalisable ($\eta = 0.0$)')
    plt.plot(m_list, agnostic_probs, marker='s', label='Cadre Agnostique ($\eta = 0.1$)')
    plt.xlabel('Taille de l\'échantillon d\'entraînement (m)')
    plt.ylabel(f'Probabilité Pr(L(h) > {epsilon})')
    plt.title('Évolution de la probabilité d\'erreur en fonction de m')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Conclusion et Synthèse ---
    print("\n" + "="*50)
    print("CONCLUSION ET SYNTHÈSE DES OBSERVATIONS")
    print("="*50)
    print("1. Cadre Réalisable (PAC) :")
    print("   - L'erreur empirique tend vers 0 lorsque m augmente.")
    print("   - La probabilité Pr(L(h) > epsilon) diminue rapidement vers 0, ce qui valide empiriquement la PAC-learnabilité du modèle.")
    print("\n2. Cadre Agnostique (Agnostic PAC) :")
    print("   - L'erreur empirique ne peut pas atteindre zéro en raison du bruit introduit (taux de bruit eta).")
    print("   - La performance de généralisation continue de s'améliorer avec m, mais se stabilise autour de la limite inhérente fixée par le bruit.")
    print("="*50)
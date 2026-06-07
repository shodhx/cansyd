class BearingRule:
    def __init__(self, severity_low=0.3, severity_high=0.6):
        """
        Symbolic Logic Layer enforcing physical consistency guardrails 
        between stochastic latent embeddings and domain physics rules.
        """
        self.sev_low = severity_low
        self.sev_high = severity_high
        
        # Hard schema definitions matching your notebook's CWRU/MFPT taxonomy
        self.fault_map = {
            0: 'Normal', 
            1: 'Ball-007', 2: 'Ball-014', 3: 'Ball-021',
            4: 'IR-007',  5: 'IR-014',  6: 'IR-021',
            7: 'OR-007',  8: 'OR-014',  9: 'OR-021'
        }

    def _quantize_energy(self, feature_norm):
        """Quantizes continuous embedding L2 spaces into physical domain states."""
        if feature_norm < self.sev_low:
            return 'Low'
        elif feature_norm < self.sev_high:
            return 'Medium'
        return 'High'
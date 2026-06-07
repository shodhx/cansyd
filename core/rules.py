# Save as core/rules.py

class BearingRule:
    def __init__(self, severity_low=0.3, severity_high=0.6):
        self.sev_low = severity_low
        self.sev_high = severity_high
        self.fault_map = {
            0: 'Normal', 1: 'Ball-007', 2: 'Ball-014', 3: 'Ball-021',
            4: 'IR-007',  5: 'IR-014',  6: 'IR-021',
            7: 'OR-007',  8: 'OR-014',  9: 'OR-021'
        }
    
    def evaluate(self, fault_type, confidence, feature_norm):
        """
        Validates the physical consistency of neural network predictions 
        against latent space energy profiles.
        """
        if feature_norm < self.sev_low:
            severity = 'Low'
        elif feature_norm < self.sev_high:
            severity = 'Medium'
        else:
            severity = 'High'
            
        # Hard Symbolic Guardrails
        is_normal_class = (fault_type == 0)
        
        # Rule 1: High energy signatures cannot map to a healthy operational state
        if is_normal_class and severity == 'High':
            return False
            
        # Rule 2: Advanced mechanical wear cannot yield low latent feature space norm
        if fault_type in [3, 6, 9] and severity == 'Low':
            return False
            
        if fault_type not in self.fault_map:
            return False
            
        return True

    def get_metadata(self, fault_type, feature_norm):
        if feature_norm < self.sev_low: severity = 'Low'
        elif feature_norm < self.sev_high: severity = 'Medium'
        else: severity = 'High'
        return {
            'fault': self.fault_map.get(fault_type, 'Unknown'),
            'severity': severity
        }

rule_engine = BearingRule()
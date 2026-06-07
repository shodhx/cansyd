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
        """Maps latent space energy profiles to check for physical contradictions."""
        if feature_norm < self.sev_low:
            severity = 'Low'
        elif feature_norm < self.sev_high:
            severity = 'Medium'
        else:
            severity = 'High'
            
        is_normal_class = (fault_type == 0)
        
        # Rule 1: High dimensional energy signatures cannot map to a healthy baseline classification
        if is_normal_class and severity == 'High':
            return False
            
        # Rule 2: Advanced mechanical wear cannot return a low structural energy norm
        if fault_type in [3, 6, 9] and severity == 'Low':
            return False
            
        if fault_type not in self.fault_map:
            return False
            
        return True

rule_engine = BearingRule()
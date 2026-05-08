class BearingRule:
    def __init__(self, severity_low=0.3, severity_high=0.6):
        self.sev_low = severity_low
        self.sev_high = severity_high
    
    def evaluate(self, fault_type, confidence, feature_norm):
        if feature_norm < self.sev_low:
            severity = 'Low'
        elif feature_norm < self.sev_high:
            severity = 'Medium'
        else:
            severity = 'High'
        
        fault_map = {0: 'Normal', 1: 'Ball-007', 2: 'Ball-014', 3: 'Ball-021',
                     4: 'IR-007', 5: 'IR-014', 6: 'IR-021',
                     7: 'OR-007', 8: 'OR-014', 9: 'OR-021'}
        
        return {'fault': fault_map.get(fault_type, 'Unknown'), 
                'severity': severity, 
                'confidence': float(confidence)}

rule_engine = BearingRule()
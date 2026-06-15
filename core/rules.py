class BearingRule:
    def __init__(self, ref_norm=15.18, low_frac=0.8, high_frac=1.2):
        """
        Symbolic Logic Layer enforcing physical consistency guardrails
        between stochastic latent embeddings and domain physics rules.

        Severity is graded *relative* to the calibrated median feature-norm
        (CAL_MEDIAN_NORM = 15.18 in the notebook). CWRU feature norms sit
        around 15, so the old absolute 0.3 / 0.6 thresholds quantised every
        sample as 'High' and vetoed every Normal prediction.
        """
        self.sev_low = ref_norm * low_frac
        self.sev_high = ref_norm * high_frac
        
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
    
    def evaluate(self, fault_type, confidence, feature_norm):
        """
        Validates structural consistency. Returns a boolean truth flag.
        Returns False if a connectionist network prediction violates physical constraints.
        """
        severity = self._quantize_energy(feature_norm)
        is_normal_class = (fault_type == 0)
        
        # Rule 1: A normal classification cannot coexist with high structural energy profiles
        if is_normal_class and severity == 'High':
            return False
            
        # Rule 2: Advanced, terminal mechanical defects cannot return ambient/low feature norms
        # Targets depths of 0.021" (labels 3, 6, 9)
        if fault_type in [3, 6, 9] and severity == 'Low':
            return False
            
        if fault_type not in self.fault_map:
            return False
            
        return True
    
    def get_metadata(self, fault_type, feature_norm):
        """Helper to compile formal report maps once structural states pass validation."""
        severity = self._quantize_energy(feature_norm)
        return {
            'fault': self.fault_map.get(fault_type, 'Unknown'),
            'severity': severity,
            'is_valid': self.evaluate(fault_type, 1.0, feature_norm)
        }

# Global module execution instance
rule_engine = BearingRule()
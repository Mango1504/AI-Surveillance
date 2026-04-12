"""
Examinee Management Module
Handles loading, validating, and managing examinee data
"""

import json
import os
from datetime import datetime

class ExamineeManager:
    def __init__(self, data_file="examinees.json"):
        """Initialize examinee manager with JSON data file"""
        self.data_file = data_file
        self.examinees = self.load_examinees()
        self.exam_start_time = datetime.now()
    
    def load_examinees(self):
        """Load examinees from JSON file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                print(f"[EXAM] Loaded {len(data)} examinees from {self.data_file}")
                return data
            else:
                print(f"[WARNING] {self.data_file} not found!")
                return []
        except Exception as e:
            print(f"[ERROR] Failed to load examinees: {e}")
            return []
    
    def get_examinees_by_hall(self, exam_hall):
        """Get all examinees for a specific exam hall"""
        return [e for e in self.examinees if e['exam_hall'] == exam_hall]
    
    def get_examinee_by_roll(self, roll_number):
        """Get examinee by roll number"""
        for examinee in self.examinees:
            if examinee['roll_number'] == roll_number:
                return examinee
        return None
    
    def is_valid_examinee_in_hall(self, exam_hall, roll_number=None):
        """
        Verify if an examinee is registered for a specific exam hall
        Returns: (is_valid, examinee_info)
        """
        if not roll_number:
            # Just check if there are valid examinees in the hall
            return len(self.get_examinees_by_hall(exam_hall)) > 0, None
        
        examinee = self.get_examinee_by_roll(roll_number)
        if examinee and examinee['exam_hall'] == exam_hall:
            return True, examinee
        return False, None
    
    def get_all_examinees(self):
        """Get all examinees"""
        return self.examinees
    
    def get_exam_hall_info(self, exam_hall):
        """Get summary info for an exam hall"""
        examinees = self.get_examinees_by_hall(exam_hall)
        return {
            "exam_hall": exam_hall,
            "total_examinees": len(examinees),
            "examinees": examinees
        }
    
    def add_examinee(self, roll_number, name, exam_hall, exam_date, subject):
        """Add a new examinee"""
        new_examinee = {
            "roll_number": roll_number,
            "name": name,
            "exam_hall": exam_hall,
            "exam_date": exam_date,
            "subject": subject
        }
        self.examinees.append(new_examinee)
        self.save_examinees()
        print(f"[EXAM] Added examinee: {name} ({roll_number})")
        return new_examinee
    
    def save_examinees(self):
        """Save examinees to JSON file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.examinees, f, indent=2)
            print(f"[EXAM] Saved {len(self.examinees)} examinees to {self.data_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save examinees: {e}")
    
    def get_stats(self):
        """Get overall statistics"""
        halls = set(e['exam_hall'] for e in self.examinees)
        return {
            "total_examinees": len(self.examinees),
            "total_halls": len(halls),
            "halls": list(halls),
            "by_hall": {hall: len(self.get_examinees_by_hall(hall)) for hall in sorted(halls)}
        }


# Singleton instance
_manager = None

def get_examinee_manager():
    """Get singleton instance of examinee manager"""
    global _manager
    if _manager is None:
        _manager = ExamineeManager()
    return _manager

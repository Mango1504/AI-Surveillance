# Examinee Verification System - Implementation Guide

## Overview

A complete examinee management system has been integrated into `second.py` that allows you to:

1. **Maintain a database** of all registered examinees
2. **Track exam halls** and their assigned examinees
3. **Log violations** when phones are detected
4. **Query examinee data** via REST API endpoints

---

## File Structure

### New Files Created

| File | Purpose |
|------|---------|
| `examinees.json` | JSON database of all registered examinees |
| `examinee_manager.py` | Python module for managing examinee data |
| `second_CORS_enabled.py` | Updated backend with examinee verification |

### Sample Data (examinees.json)

```json
{
  "roll_number": "001",
  "name": "John Smith",
  "exam_hall": 1,
  "exam_date": "2024-04-12",
  "subject": "Mathematics"
}
```

**Fields:**
- `roll_number` - Unique student ID
- `name` - Student name
- `exam_hall` - Assigned exam hall (1-4)
- `exam_date` - Exam date
- `subject` - Subject being examined

---

## 📊 API Endpoints for Examinee Data

### 1. Get All Examinees
```
GET /examinees
```

**Response:**
```json
{
  "total": 8,
  "examinees": [...],
  "stats": {
    "total_examinees": 8,
    "total_halls": 4,
    "halls": [1, 2, 3, 4],
    "by_hall": {
      "1": 2,
      "2": 2,
      "3": 2,
      "4": 2
    }
  }
}
```

### 2. Get Examinees for Specific Hall
```
GET /examinees/hall/<hall_number>
```

Example: `/examinees/hall/1`

**Response:**
```json
{
  "exam_hall": 1,
  "total": 2,
  "examinees": [
    {
      "roll_number": "001",
      "name": "John Smith",
      "exam_hall": 1,
      "exam_date": "2024-04-12",
      "subject": "Mathematics"
    }
  ]
}
```

### 3. Get All Violations
```
GET /violations
```

**Response:**
```json
{
  "total": 5,
  "violations": [
    {
      "timestamp": "2024-04-12T10:30:45Z",
      "exam_hall": 2,
      "grid_position": "R2C3",
      "confidence": 0.95,
      "status": "VIOLATION - Unauthorized Phone"
    }
  ]
}
```

### 4. Get Hall Information with Violations
```
GET /hall-info/<hall_number>
```

Example: `/hall-info/2`

**Response:**
```json
{
  "exam_hall": 2,
  "total_examinees": 2,
  "examinees": [...],
  "violations": 3,
  "violation_details": [
    {
      "timestamp": "2024-04-12T10:30:45Z",
      "exam_hall": 2,
      "grid_position": "R2C3",
      "confidence": 0.95,
      "status": "VIOLATION - Unauthorized Phone"
    }
  ]
}
```

---

## 🔧 Using the ExamineeManager Module

### In Python Code

```python
from examinee_manager import get_examinee_manager

# Get the singleton manager instance
manager = get_examinee_manager()

# Get all examinees
all_examinees = manager.get_all_examinees()

# Get examinees for a hall
hall_1_examinees = manager.get_examinees_by_hall(1)

# Get specific examinee
examinee = manager.get_examinee_by_roll("001")

# Check if valid examinee in hall
is_valid, examinee_info = manager.is_valid_examinee_in_hall(1, "001")

# Get hall summary
hall_info = manager.get_exam_hall_info(1)

# Get statistics
stats = manager.get_stats()

# Add new examinee
manager.add_examinee("009", "New Student", 2, "2024-04-12", "Biology")

# Save to file
manager.save_examinees()
```

---

## 📝 Managing Examinee Data

### Add Examinees

You can add examinees in two ways:

#### Method 1: Edit JSON File Directly

Edit `examinees.json`:

```json
[
  {
    "roll_number": "001",
    "name": "John Smith",
    "exam_hall": 1,
    "exam_date": "2024-04-12",
    "subject": "Mathematics"
  },
  {
    "roll_number": "009",
    "name": "New Student",
    "exam_hall": 2,
    "exam_date": "2024-04-12",
    "subject": "Biology"
  }
]
```

#### Method 2: Use Python Script

Create `add_examinee.py`:

```python
from examinee_manager import get_examinee_manager

manager = get_examinee_manager()

# Add multiple examinees
manager.add_examinee("009", "New Student", 2, "2024-04-12", "Biology")
manager.add_examinee("010", "Jane Doe", 3, "2024-04-12", "Physics")

# Save changes
manager.save_examinees()

print("Examinees added successfully!")
```

Run:
```bash
python add_examinee.py
```

### Update Examinees

Edit `examinees.json` directly and run:

```python
from examinee_manager import get_examinee_manager
manager = get_examinee_manager()
manager.save_examinees()
```

### Delete Examinees

Remove entries from `examinees.json` and save.

---

## 📋 Violation Logging

### How It Works

When a phone is detected:
1. System logs it to the violation logger
2. Creates a violation record with:
   - Timestamp
   - Exam hall
   - Grid position (R2C3)
   - Confidence score
   - Status

### Accessing Violations

**Via API:**
```bash
curl http://localhost:5000/violations
```

**Via Python:**
```python
from examinee_manager import get_examinee_manager

# In second.py, access via:
violation_logger.get_violations()
```

### Violation Log File

Violations are also logged to:
```
C:/Users/write/Desktop/Phone/violations.log
```

Format:
```
[2024-04-12T10:30:45Z] ExamHall 2 - Phone at R2C3 (Conf: 0.95)
[2024-04-12T10:35:12Z] ExamHall 1 - Phone at R1C2 (Conf: 0.87)
```

---

## 🔍 Integration with React Frontend

You can integrate this data into your React app by adding API calls:

```javascript
// src/services/api.js

export const apiService = {
  // ... existing methods ...
  
  // Examinee endpoints
  getExaminees: async () => {
    const response = await api.get('/examinees')
    return response.data
  },
  
  getExamineesHall: async (hallNumber) => {
    const response = await api.get(`/examinees/hall/${hallNumber}`)
    return response.data
  },
  
  getHallInfo: async (hallNumber) => {
    const response = await api.get(`/hall-info/${hallNumber}`)
    return response.data
  },
  
  getViolations: async () => {
    const response = await api.get('/violations')
    return response.data
  },
}
```

### Create Violations Component

```javascript
// src/components/ViolationsTab.jsx

import { useEffect, useState } from 'react'
import { apiService } from '../services/api'

export default function ViolationsTab() {
  const [violations, setViolations] = useState([])
  
  useEffect(() => {
    const fetchViolations = async () => {
      const data = await apiService.getViolations()
      setViolations(data.violations)
    }
    
    const interval = setInterval(fetchViolations, 2000)
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-white">Phone Violations</h2>
      {violations.length === 0 ? (
        <p className="text-gray-400">No violations detected</p>
      ) : (
        violations.map((v, i) => (
          <div key={i} className="bg-red-900 p-4 rounded">
            <p><strong>Hall:</strong> {v.exam_hall}</p>
            <p><strong>Position:</strong> {v.grid_position}</p>
            <p><strong>Confidence:</strong> {(v.confidence * 100).toFixed(1)}%</p>
            <p><strong>Time:</strong> {v.timestamp}</p>
          </div>
        ))
      )}
    </div>
  )
}
```

---

## 🚀 Startup Output

When you start the backend, you'll see:

```
[CAM] 640x480  grid=3x4
[EXAM] Total Examinees: 8
       ExamHall 1: 2 examinees
       ExamHall 2: 2 examinees
       ExamHall 3: 2 examinees
       ExamHall 4: 2 examinees

[API] Running at http://0.0.0.0:5000
      /status      → JSON detection state
      /stream      → MJPEG live feed
      /snapshot    → single JPEG
      /grid-info   → grid dimensions

[EXAMINEES]
      /examinees        → all registered examinees
      /examinees/hall/<n> → examinees for hall N
      /hall-info/<n>    → hall info with examinees & violations

[VIOLATIONS]
      /violations  → all logged violations

[CORS] Enabled for React frontend at http://localhost:3000
```

---

## 📊 Example Workflows

### Workflow 1: Check if Student is Authentic

```python
hall_number = 1
roll_number = "001"

is_valid, examinee_info = manager.is_valid_examinee_in_hall(
    hall_number, 
    roll_number
)

if is_valid:
    print(f"Valid: {examinee_info['name']}")
else:
    print("UNAUTHORIZED - Violator Detected!")
    violation_logger.log_violation(hall_number, row, col, confidence, timestamp)
```

### Workflow 2: Export Violations Report

```python
violations = violation_logger.get_violations()

# Group by hall
by_hall = {}
for v in violations:
    hall = v['exam_hall']
    if hall not in by_hall:
        by_hall[hall] = []
    by_hall[hall].append(v)

# Print report
for hall, hall_violations in by_hall.items():
    print(f"\nExam Hall {hall}: {len(hall_violations)} violations")
    for v in hall_violations:
        print(f"  {v['timestamp']} - {v['grid_position']}")
```

---

## ⚙️ Configuration

### Change Exam Hall Count

Edit `second_CORS_enabled.py`:
```python
GRID_COLS = 4  # Change if needed
```

### Add More Fields to Examinee

Edit `examinees.json`:
```json
{
  "roll_number": "001",
  "name": "John Smith",
  "exam_hall": 1,
  "exam_date": "2024-04-12",
  "subject": "Mathematics",
  "seat_number": "A1",
  "batch": "2024-A",
  "phone_number": "1234567890"
}
```

---

## 🔒 Security Tips

1. **Backup examinees.json** regularly
2. **Validate input** before adding examinees
3. **Monitor violations.log** for unauthorized access attempts
4. **Restrict API access** in production (use authentication)
5. **Encrypt roll numbers** for privacy

---

## 📞 Testing

### Test with curl

```bash
# Get all examinees
curl http://localhost:5000/examinees

# Get hall 1 examinees
curl http://localhost:5000/examinees/hall/1

# Get violations
curl http://localhost:5000/violations

# Get complete hall info
curl http://localhost:5000/hall-info/2
```

### Test in Python

```python
import requests

# Get all examinees
resp = requests.get('http://localhost:5000/examinees')
print(resp.json())

# Get violations
resp = requests.get('http://localhost:5000/violations')
print(resp.json())
```

---

## 🎯 Next Steps

1. ✅ Files created and integrated
2. ✅ API endpoints available
3. 📝 **Edit `examinees.json`** with your actual examinee data
4. 🚀 **Start the backend**: `python second_CORS_enabled.py`
5. 🌐 **Access APIs** as shown above
6. 🔌 **Integrate with React frontend** (optional)

---

## 📚 Full Example: Complete Integration

```python
# Example: Query system for multiple halls

from examinee_manager import get_examinee_manager

manager = get_examinee_manager()

# Get system stats
stats = manager.get_stats()
print(f"Total examinees: {stats['total_examinees']}")

# Check each hall
for hall in sorted(stats['halls']):
    hall_info = manager.get_exam_hall_info(hall)
    print(f"\nHall {hall}:")
    print(f"  Examinees: {hall_info['total_examinees']}")
    for examinee in hall_info['examinees']:
        print(f"    - {examinee['roll_number']}: {examinee['name']}")
```

---

**System is now ready for examinee verification! 🎉**

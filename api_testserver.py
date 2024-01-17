from flask import Flask, request, jsonify
from datetime import datetime
import uuid

app = Flask(__name__)

# Mock data for testing
devices = {}
events = {}

# Endpoint for getting device task
@app.route('/api/v1/device/task', methods=['POST'])
def get_device_task():
    device_id = request.json.get('device_id', '')
    # Here you can add logic to return specific tasks for the device
    return jsonify({"status": 1, "task": "Sample task for device " + device_id})

# Endpoint for sending heartbeat
@app.route('/api/v1/device/heartbeat', methods=['POST'])
def send_heartbeat():
    data = request.json
    data['received_at'] = datetime.now().isoformat()
    # Storing the heartbeat data for testing purposes
    devices[data['device_id']] = data
    return jsonify({"status": 1, "message": "Heartbeat received"})

# Endpoint for sending falling rock event
@app.route('/api/v1/event/add', methods=['POST'])
def send_event():
    event_data = request.json
    if event_data['type'] == 'falling_rock':
        # Store or process the falling rock event data
        event_id = str(uuid.uuid4())
        events[event_id] = event_data
        return jsonify({"status": 1, "message": "Falling rock event received", "event_id": event_id})
    elif event_data['type'] == 'surface_change':
        # Store or process the surface change event data
        event_id = str(uuid.uuid4())
        events[event_id] = event_data
        return jsonify({"status": 1, "message": "Surface change event received", "event_id": event_id})
    else:
        return jsonify({"status": 0, "message": "Unknown event type"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

// Carbon Sentinel MVP: Arduino UNO data acquisition sketch
//
// This sketch reads one analog channel from an rGO strain sensor wired as a
// voltage divider and sends the estimated resistance over Serial.
//
// Expected wiring:
//   5V -> fixed resistor -> A0 -> rGO sensor -> GND
//
// Serial output:
//   One numeric resistance value per line, in ohms.

const int sensorPin = A0;
const float vcc = 5.0;
const float adcMax = 1023.0;
const float seriesResistance = 10000.0;
const unsigned long sampleDelayMs = 100;

void setup() {
  Serial.begin(9600);
  pinMode(sensorPin, INPUT);
}

float readSensorResistance() {
  int adcValue = analogRead(sensorPin);

  if (adcValue <= 0) {
    return 0.0;
  }

  if (adcValue >= adcMax) {
    return -1.0;
  }

  float voltage = (adcValue / adcMax) * vcc;

  if (voltage <= 0.0 || voltage >= vcc) {
    return -1.0;
  }

  return seriesResistance * (voltage / (vcc - voltage));
}

void loop() {
  float resistance = readSensorResistance();

  if (resistance >= 0.0) {
    Serial.println(resistance, 6);
  } else {
    Serial.println(0.0, 6);
  }

  delay(sampleDelayMs);
}
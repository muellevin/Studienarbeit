float horizontalValue = 0.0;
float verticalValue = 0.0;
bool commandReceived = false;

#include <Servo.h>

int horizontalServoPin = 7;
Servo horizontalServo;
float horizontalServoPos = 90;

int verticalServoPin = 6;
Servo verticalServo;
float verticalServoPos = 90;

int powerPumpPin = 9;

int powerServoPin = 11;

int powerLightPin = 10;

// int soundSignalPin = 3;
// unsigned long currentSoundFrequency = 15000;
// unsigned long UpperSoundFrequency = 44000;
// unsigned long lowerSoundFrequency = 15000;
// int frequencyStepSize = 100;
// pin 12 is broken
// vertical servo is broken
int powerSoundPin = 12;

void setup() {
  Serial.begin(115200);
  pinMode(powerServoPin, OUTPUT);
  pinMode(powerLightPin, OUTPUT);
  pinMode(powerPumpPin, OUTPUT);
  pinMode(powerSoundPin, OUTPUT);
  horizontalServo.attach(horizontalServoPin);
  // verticalServo.attach(verticalServoPin);
  delay(100);
  horizontalServo.detach();
  // verticalServo.detach();
}

bool initialized = false;
void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'h' || command == 'v') {

      float value = Serial.parseFloat();

      if (command == 'h') {
        horizontalValue = 90 + value;
      } else if (command == 'v') {
        verticalValue = 90 + value;
      }
      if (!initialized) {
        horizontalServo.attach(horizontalServoPin);
        // verticalServo.attach(verticalServoPin);
        initialized = true;
      }
      commandReceived = true;
    } else if (command == 'g') {
      // Go in Standby mode
      initialized = false;
      commandReceived = false;
      horizontalServo.detach();
      // verticalServo.detach();
    }
  }

  if (commandReceived) {

    digitalWrite(powerServoPin, HIGH);
    digitalWrite(powerPumpPin, HIGH);
    // verticalServo.write((int)verticalValue);
    horizontalServo.write((int)horizontalValue);
    // moveServo(horizontalServo, 20, 160, &horizontalValue, &horizontalServoPos);
    // moveServo(verticalServo, 0, 45, &verticalValue, &verticalServoPos);
    if (millis() % 200 == 0) {  // 5H
      digitalWrite(powerLightPin, !digitalRead(powerLightPin));
    }
    if (millis() % 1000 == 0) {  // 1H
      {
        digitalWrite(powerSoundPin, !digitalRead(powerSoundPin));
      }
      // if (millis() % 100 == 0) {
      //   if (currentSoundFrequency < lowerSoundFrequency) {
      //     currentSoundFrequency = lowerSoundFrequency;
      //     frequencyStepSize *= -1;
      //   } else if (currentSoundFrequency > UpperSoundFrequency) {
      //     currentSoundFrequency = UpperSoundFrequency;
      //     frequencyStepSize *= -1;
      //   }
      //   tone(soundSignalPin, currentSoundFrequency);
      //   currentSoundFrequency += frequencyStepSize;
      //   Serial.print("Playing sound with frequency: ");
      //   Serial.println(currentSoundFrequency); // Printing the frequency of the sound played
      // }
    }
  } else {

    digitalWrite(powerServoPin, LOW);
    digitalWrite(powerPumpPin, LOW);
    digitalWrite(powerLightPin, LOW);
    digitalWrite(powerSoundPin, LOW);
  }
}

void moveServo(Servo motor, uint8_t minPos, uint8_t maxPos, float *targetPos, float *currentPosition) {
  if (*targetPos < minPos) {
    *currentPosition = minPos;
  } else if (*targetPos > maxPos) {
    *currentPosition = maxPos;
  } else {
    *currentPosition = *targetPos;
  }
  Serial.println((String) "pos: " + *currentPosition);
  motor.write(verticalValue);
}

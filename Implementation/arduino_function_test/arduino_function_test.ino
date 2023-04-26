#include <Servo.h>

int horizontalServoPin = 10;
Servo horizontalServo;
float horizontalServoPos = 90;
float horizontalServoStepSize = 20;

int verticalServoPin = 9;
Servo verticalServo;
float verticalServoPos = 90;
float verticalServoStepSize = 10;

int powerServoPin = 7;

int powerLightPin = 6;

int powerPumpPin = 4;

int soundSignalPin = 3;
unsigned long currentSoundFrequency = 15000;
unsigned long UpperSoundFrequency = 44000;
unsigned long lowerSoundFrequency = 15000;
int frequencyStepSize = 100;

int powerSoundPin = 2;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  horizontalServo.attach(horizontalServoPin);
  verticalServo.attach(verticalServoPin);
  pinMode(powerServoPin, OUTPUT);
  pinMode(powerLightPin, OUTPUT);
  pinMode(powerPumpPin, OUTPUT);
  pinMode(powerSoundPin, OUTPUT);
  pinMode(soundSignalPin, OUTPUT);
  digitalWrite(powerServoPin, HIGH);
}

void loop() {
  if (millis() % 500 == 0) {
    moveSevo(horizontalServo, 0, 180, &horizontalServoStepSize, &horizontalServoPos);
    Serial.print("Moving horizontal Servo to position ");
    Serial.println(horizontalServoPos); // Printing the position of the horizontal servo
  }
  if (millis() % 1000 == 0) {
    moveSevo(verticalServo, 45, 135, &verticalServoStepSize, &verticalServoPos);
    Serial.print("Moving vertical Servo to position ");
    Serial.println(verticalServoPos); // Printing the position of the vertical servo
  }
  if (millis() % 200 == 0) {
    digitalWrite(powerLightPin, !digitalRead(powerLightPin));
    Serial.println("Toggling powerLightPin."); // A message indicating powerLightPin has been toggled
  }
  // if (millis() % 5000 == 0) {
  //   digitalWrite(powerServoPin, !digitalRead(powerServoPin));
  //   Serial.println("Toggling powerServoPin."); // A message indicating powerServoPin has been toggled
  // }
  // if (millis() % 1000 == 0) {
  //   digitalWrite(powerPumpPin, !digitalRead(powerPumpPin));
  //   Serial.println("Toggling powerPumpPin."); // A message indicating powerPumpPin has been toggled
  // }
  if (millis() % 10000 == 0) {
    digitalWrite(powerSoundPin, !digitalRead(powerSoundPin));
    Serial.println("Toggling powerSoundPin."); // A message indicating powerSoundPin has been toggled
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


void moveSevo(Servo motor, uint8_t minPos, uint8_t maxPos, float* stepSize, float* currentPosition) {
  if (*currentPosition < minPos) {
    *currentPosition = minPos;
    *stepSize *= -1;
  } else if (*currentPosition > maxPos) {
    *currentPosition = maxPos;
    *stepSize *= -1;
  }
  motor.write(*currentPosition);
  *currentPosition += *stepSize;
}
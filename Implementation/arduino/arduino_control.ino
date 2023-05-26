float horizontalValue = 0.0;
float verticalValue = 0.0;
bool commandReceived = false;

#include <Servo.h>

int horizontalServoPin = 10;
Servo horizontalServo;
float horizontalServoPos = 90;

int verticalServoPin = 9;
Servo verticalServo;
float verticalServoPos = 90;

int powerServoPin = 7;

int powerLightPin = 6;

int powerPumpPin = 4;

// int soundSignalPin = 3;
// unsigned long currentSoundFrequency = 15000;
// unsigned long UpperSoundFrequency = 44000;
// unsigned long lowerSoundFrequency = 15000;
// int frequencyStepSize = 100;

int powerSoundPin = 9;

void setup()
{
    Serial.begin(115200);
    pinMode(powerServoPin, OUTPUT);
    pinMode(powerLightPin, OUTPUT);
    pinMode(powerPumpPin, OUTPUT);
    pinMode(powerSoundPin, OUTPUT);
}

void loop()
{
    if (Serial.available() > 0)
    {
        char command = Serial.read();

        if (command == 'h' || command == 'v')
        {

            float value = Serial.parseFloat();

            if (command == 'h')
            {
                horizontalValue = value;
            }
            else if (command == 'v')
            {
                verticalValue = value;
            }
            if (commandReceived)
            {
                horizontalServo.attach(horizontalServoPin);
                verticalServo.attach(verticalServoPin);
            }
            commandReceived = true;
        }
        else if (command == 'g')
        {
            // Go in Standby mode
            commandReceived = false;
            horizontalServo.detach(horizontalServoPin);
            verticalServo.detach(verticalServoPin);
        }
    }

    if (commandReceived)
    {

        digitalWrite(powerServoPin, HIGH);
        digitalWrite(powerPumpPin, HIGH);
        moveServo(horizontalServo, 50, 130, &horizontalValue, &horizontalServoPos);
        moveServo(verticalServo, 45, 135, &verticalValue, &verticalServoPos);
        if (millis() % 200 == 0)
        { // 5H
            digitalWrite(powerLightPin, !digitalRead(powerLightPin));
        }
        if (millis() % 1000 == 0)
        { // 1H
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
    }
    else
    {

        digitalWrite(powerServoPin, LOW);
        digitalWrite(powerPumpPin, LOW);
        digitalWrite(powerLightPin, LOW);
        digitalWrite(powerSoundPin, LOW);
    }
}

void moveServo(Servo motor, uint8_t minPos, uint8_t maxPos, float *targetPos, float *currentPosition)
{
    if (*targetPos < minPos)
    {
        *currentPosition = minPos;
    }
    else if (*targetPos > maxPos)
    {
        *currentPosition = maxPos;
    }
    else
    {
        *currentPosition = *targetPos;
    }
    motor.write(*currentPosition);
}

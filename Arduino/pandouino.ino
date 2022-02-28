int pinEngine = 7;
int pinWarning = 6;
int pinAccelerationLimit = 5;
int pinBrake = 4;
int pinSound = 8;

boolean engine = false;

char command = 0;

void Exit(){
  engine = false;
  digitalWrite(pinEngine,LOW);
  digitalWrite(pinWarning,LOW);
  digitalWrite(pinAccelerationLimit,LOW);
  digitalWrite(pinBrake,LOW);
  
  noTone(pinSound);
}

void Engine(){
  engine = true;
  digitalWrite(pinEngine, HIGH);
}

void Warning(){
  digitalWrite(pinWarning, HIGH);
}

void StopWarning(){
  digitalWrite(pinWarning, LOW);
}

void AccelerationLimit(){
  digitalWrite(pinAccelerationLimit, HIGH);
}

void StopAccelerationLimit(){
  digitalWrite(pinAccelerationLimit, LOW);
}

void Brake(){
  digitalWrite(pinBrake, HIGH);
}

void StopBrake(){
  digitalWrite(pinBrake, LOW);
}

void Sound(){
  tone(pinSound, 666);
}

void StopSound(){
  noTone(pinSound);
}

void setup()
{
  pinMode(pinEngine, OUTPUT);
  pinMode(pinWarning,OUTPUT);
  pinMode(pinAccelerationLimit,OUTPUT);
  pinMode(pinBrake,OUTPUT);
  
  Serial.begin(9600);
  while(!Serial){} 
  digitalWrite(pinEngine, HIGH);
  Warning();
  AccelerationLimit();
  Brake();
  Sound();
  
  delay(500);
  
  digitalWrite(pinEngine, LOW);
  StopWarning();
  StopAccelerationLimit();
  StopBrake();
  StopSound();
}

void loop()
{
  if(!engine)
  {
    command = Serial.read();
    if(command == 'e')
      Engine();
    command = 0;
  }
  else
  {
    if(Serial.available() > 0) 
    {
      command = Serial.read();
      switch(command){
        case 'x':
          Exit();
        break;
        case 'w':
          Warning();
        break;
        case 'r':
          StopWarning();
        break;
        case 'b':
          Brake();
        break;
        case 'n':
          StopBrake();
        break;
        case 'a':
          AccelerationLimit();
        break;
        case 'q':
          StopAccelerationLimit();
        break;
        case 's':
          Sound();
        break;
        case 'd':
          StopSound();
        break;
      }
      command = 0;
    }
  }
}

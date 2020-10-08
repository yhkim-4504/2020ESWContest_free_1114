#include <SPI.h>
#include <MFRC522.h>


#define SDA_PIN 10   // SDA핀 설정
#define RST_PIN 9   // RESET 핀 설정
#define TRIG 5
#define ECHO 3
#define True 1
#define False 0

long de=0;

const int stepPin = 6;
const int dirPin = 7;

int doorstate = 1;
int ST = 1;
long dis_mm;
int st_1 = 1;
unsigned long preTime;
unsigned long sensorTime;
int tag = 1;
char ch = 0;
int Number;
String content= "";

void sensor1();
void sensor2();
void opendoor();
void closedoor();
long distance(long time);
char readSerial();
long sonic_sensor();

MFRC522 rfid(SDA_PIN, RST_PIN);   // rfid로 객체 생성

void setup()

{
  
  Serial.begin(9600);   // 통신 속도 9600으로 시리얼 통신 시작
  
  SPI.begin();      // SPI 통신 시작
  
  rfid.PCD_Init();   // RFID 시작
  
  Serial.println("touch a key");
  
  Serial.println();
  pinMode(TRIG, OUTPUT);  
  pinMode(ECHO, INPUT);
  pinMode(2, OUTPUT);
}

void loop() {
 
 readSerial();
 Serial.print("ch : ");
 Serial.println(ch);
 long distance;
 distance = sonic_sensor();
 Serial.print("distance : ");
 Serial.println(distance);

 delay(200);
 if(tag==1)
 {
  sensor2();  // 카드
 }
 
 if(distance < 40)
 {
    ST = 0;
 }
 else
 {
    ST = 1;
 }
Serial.print("doorstate : ");
Serial.println(doorstate);
Serial.print("tag : ");
Serial.println(tag);
 unsigned long nowTime = millis(); //현재시각 

 if(doorstate == 1 )        //문이 닫혀있을때
 {
    if(tag == 0 && ch == 'y')
    {
      
        preTime = millis(); // 카드 찍은 시각
        opendoor();
        doorstate = 0;
        ch = 0;
        
    }
    else if(tag == 0 && ch == 'n')
    {
      tag = 1;
      ch = 0;
      digitalWrite(2, LOW);
    }
 }
 else if(doorstate == 0)  //문이 열려있을때
 {
      if(nowTime - preTime > 10000 && ST == 1)
      {
          closedoor();
          doorstate = 1;
          tag = 1;
          digitalWrite(2, LOW);
      }
      else if(ST == 0)
      {
          st_1 =0;
          sensorTime = millis();
      }
      else if(st_1 == 0)
      {
          if(nowTime - sensorTime > 2000 && ST == 1) // 초음파 센서에 사람이 잡힌 후 2초이상 경과했을때
          {
              closedoor();
              doorstate = 1;
              st_1 = 1;
              tag = 1;
              digitalWrite(2, LOW);
          }
      }
 }

}


void sensor2()  // 카드 
{
     int con;
       
       if ( ! rfid.PICC_IsNewCardPresent())
    
     {
       return;
     }
    
     // ID가 읽어지면 진행, 읽지못하면 리턴
    
     if ( ! rfid.PICC_ReadCardSerial())
    
     {
       return;
     }
    
     Serial.print("UID tag :");
    
     String content= ""; // 문자열 자료형 content 선언 
    
     for (byte i = 0; i < rfid.uid.size; i++) // tag를 읽고 출력하는 프로그램
    
     {
    
       // 삼항 연산자. 16(0x10)보다 작으면 " 0"을 아니면 " "을 출력
    
       Serial.print(rfid.uid.uidByte[i] < 0x10 ? " 0" : " ");
    
       // 16진수로 출력
    
       Serial.print(rfid.uid.uidByte[i], HEX);
    
       // 문자열을 string에 추가
    
       content.concat(String(rfid.uid.uidByte[i] < 0x10 ? " 0" : " "));
       content.concat(String(rfid.uid.uidByte[i], HEX));
     }
    
     Serial.println();
     Serial.print("Message : ");
     content.toUpperCase(); // string의 문자를 대문자로 수정
     
      if (content.substring(1) == "0E 81 DB 90")

 {

   tag =0;
   digitalWrite(2, HIGH);
   Serial.println("접근 승인-주황"); // 메시지 출력

   Serial.println("orange");
  
  

 }
 if (content.substring(1) == "FB 3D 29 83")

 {

   tag =0;
   digitalWrite(2, HIGH);
   Serial.println("접근 승인-분홍"); // 메시지 출력

   Serial.println("pink");

   

 }

 if (content.substring(1) == "FA 39 FA 80")

 {
   tag =0;
   digitalWrite(2, HIGH);
   Serial.println("접근 승인-파랑"); // 메시지 출력

   Serial.println("blue");



 }
  if (content.substring(1) == "66 C2 12 29")

 {
   tag =0;
   digitalWrite(2, HIGH);
   Serial.println("접근 승인-하양"); // 메시지 출력

   Serial.println("white");



 }

 else   

 {

   Serial.println("접근 불가"); // 메시지 출력


 }

}


void opendoor() // 문열기
{
    digitalWrite(dirPin, HIGH); // 모터를 특정한 방향으로 설정합니다.
  
    for (int x = 0; x < 2000; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
    }
 
}

void closedoor() //문닫기
{
      digitalWrite(dirPin, LOW); // 모터를 특정한 방향으로 설정합니다.

    for (int x = 0; x < 2000; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
    }

}

long distance(long time)
{
  long cal;
  cal=(((time*340)/1000)/2);
  return cal;
}

char readSerial()
{
  
  if(Serial.available())
  {
    ch = Serial.read();
  }
  return ch;

}

long sonic_sensor()
{
  long duration, distance;
  
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  duration = pulseIn (ECHO, HIGH); //물체에 반사되어돌아온 초음파의 시간을 변수에 저장합니다.
  distance = duration * 17 / 1000;
  
  return distance;
}

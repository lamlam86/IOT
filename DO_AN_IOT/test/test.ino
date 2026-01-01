int cambien = A3;
int giatri;

void setup(){
  Serial.begin(9600);
  pinMode(cambien, INPUT);
}

void loop(){
  giatri = analogRead(cambien);
  Serial.print("Gia tri cam bien: ");
  Serial.println(giatri);
  delay(200);
}
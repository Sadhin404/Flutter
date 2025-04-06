abstract class Vehicle{
    int _speed=0;
    void move();
    
 void setSpeed (int speed){
     _speed=speed;
 }
 int get speedvalue => _speed;
}

class Car extends Vehicle{
    @override
    void move (){
        print ("The car is moving at $speedvalue km/h ");
    }
}

main (){
    Car newCar = Car();
    newCar.setSpeed(30);
    newCar.move();
}

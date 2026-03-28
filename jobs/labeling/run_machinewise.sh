cd ../..
source venv/bin/activate

for machine in "fan"  "pump"  "slider"  "ToyCar"  "ToyConveyor"  "valve"; do
python -m asdkit.bin.label experiments="dcase2020_machinewise" +machine=$machine
done

for machine in "fan"  "gearbox"  "pump"  "slider"  "ToyCar"  "ToyTrain"  "valve"; do
python -m asdkit.bin.label experiments="dcase2021_machinewise" +machine=$machine
done

for machine in "bearing"  "fan"  "gearbox"  "slider"  "ToyCar"  "ToyTrain"  "valve"; do
python -m asdkit.bin.label experiments="dcase2022_machinewise" +machine=$machine
done

for machine in "bandsaw" "bearing" "fan" "gearbox" "grinder" "shaker" "slider" "ToyCar" "ToyDrone" "ToyNscale" "ToyTank" "ToyTrain"  "Vacuum" "valve"; do
python -m asdkit.bin.label experiments="dcase2023_machinewise" +machine=$machine
done

for machine in "3DPrinter" "AirCompressor" "bearing" "BrushlessMotor" "fan" "gearbox" "HairDryer" "HoveringDrone" "RoboticArm" "Scanner" "slider" "ToothBrush" "ToyCar" "ToyCircuit" "ToyTrain" "valve"; do
python -m asdkit.bin.label experiments="dcase2024_machinewise" +machine=$machine
done

for machine in "AutoTrash" "BandSealer" "bearing" "CoffeeGrinder" "fan" "gearbox" "HomeCamera" "Polisher" "ScrewFeeder" "slider" "ToyCar" "ToyPet" "ToyRCCar" "ToyTrain" "valve"; do
python -m asdkit.bin.label experiments="dcase2025_machinewise" +machine=$machine
done

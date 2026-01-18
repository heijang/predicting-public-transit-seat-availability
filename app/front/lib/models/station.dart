import 'train_arrival.dart';

class Station {
  final String stopId;
  final String name;
  final int orderIndex;
  final List<TrainArrival> arrivals;

  const Station({
    required this.stopId,
    required this.name,
    required this.orderIndex,
    this.arrivals = const [],
  });

  Station copyWith({
    String? stopId,
    String? name,
    int? orderIndex,
    List<TrainArrival>? arrivals,
  }) {
    return Station(
      stopId: stopId ?? this.stopId,
      name: name ?? this.name,
      orderIndex: orderIndex ?? this.orderIndex,
      arrivals: arrivals ?? this.arrivals,
    );
  }
}

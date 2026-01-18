import 'package:flutter/material.dart';
import '../models/station.dart';
import '../data/station_data.dart';

class LineMapView extends StatelessWidget {
  final List<Station> stations;
  final Station? departureStation;
  final Station? arrivalStation;
  final ValueChanged<Station> onStationTap;

  const LineMapView({
    super.key,
    required this.stations,
    this.departureStation,
    this.arrivalStation,
    required this.onStationTap,
  });

  bool _isInRoute(Station station) {
    if (departureStation == null || arrivalStation == null) return false;

    final depIndex = departureStation!.orderIndex;
    final arrIndex = arrivalStation!.orderIndex;
    final stationIndex = station.orderIndex;

    if (depIndex <= arrIndex) {
      return stationIndex >= depIndex && stationIndex <= arrIndex;
    } else {
      return stationIndex >= arrIndex && stationIndex <= depIndex;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 4,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 14,
                height: 14,
                decoration: BoxDecoration(
                  color: StationData.lineColor,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                StationData.lineName,
                style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 55,
            child: LayoutBuilder(
              builder: (context, constraints) {
                final itemWidth = constraints.maxWidth / stations.length;
                return Stack(
                  children: [
                    Positioned(
                      left: itemWidth / 2,
                      right: itemWidth / 2,
                      top: 10,
                      child: Container(
                        height: 4,
                        decoration: BoxDecoration(
                          color: Colors.grey[300],
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                    if (departureStation != null && arrivalStation != null)
                      Positioned(
                        left: itemWidth / 2 +
                            (departureStation!.orderIndex <
                                        arrivalStation!.orderIndex
                                    ? departureStation!.orderIndex
                                    : arrivalStation!.orderIndex) *
                                itemWidth,
                        top: 10,
                        child: Container(
                          width:
                              ((departureStation!.orderIndex -
                                              arrivalStation!.orderIndex)
                                          .abs() *
                                      itemWidth),
                          height: 4,
                          decoration: BoxDecoration(
                            color: StationData.lineColor,
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                      ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: stations.map((station) {
                        final isSelected =
                            station.stopId == departureStation?.stopId ||
                                station.stopId == arrivalStation?.stopId;
                        final isInRoute = _isInRoute(station);

                        return Expanded(
                          child: GestureDetector(
                            onTap: () => onStationTap(station),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Container(
                                  width: isSelected ? 18 : 12,
                                  height: isSelected ? 18 : 12,
                                  decoration: BoxDecoration(
                                    color: isInRoute || isSelected
                                        ? StationData.lineColor
                                        : Colors.white,
                                    shape: BoxShape.circle,
                                    border: Border.all(
                                      color: isInRoute
                                          ? StationData.lineColor
                                          : Colors.grey[400]!,
                                      width: 1.5,
                                    ),
                                    boxShadow: isSelected
                                        ? [
                                            BoxShadow(
                                              color: StationData.lineColor
                                                  .withValues(alpha: 0.4),
                                              blurRadius: 3,
                                              spreadRadius: 0.5,
                                            ),
                                          ]
                                        : null,
                                  ),
                                  child: isSelected
                                      ? const Icon(Icons.check,
                                          color: Colors.white, size: 10)
                                      : null,
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  station.name.length > 8
                                      ? '${station.name.substring(0, 7)}.'
                                      : station.name,
                                  style: TextStyle(
                                    fontSize: 7,
                                    fontWeight: isSelected
                                        ? FontWeight.bold
                                        : FontWeight.normal,
                                    color: isInRoute
                                        ? StationData.lineColor
                                        : Colors.grey[700],
                                  ),
                                  textAlign: TextAlign.center,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

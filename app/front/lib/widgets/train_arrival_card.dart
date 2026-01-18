import 'package:flutter/material.dart';
import '../models/train_arrival.dart';
import 'congestion_badge.dart';

class TrainArrivalCard extends StatelessWidget {
  final TrainArrival arrival;
  final int index;

  const TrainArrivalCard({
    super.key,
    required this.arrival,
    required this.index,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 3,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              color: _getIndexColor(index),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                '${index + 1}',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 11,
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      arrival.arrivalText,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(width: 6),
                    CongestionBadge(congestion: arrival.congestion),
                  ],
                ),
                Text(
                  '→ ${arrival.destination} (${arrival.trainNumber})',
                  style: TextStyle(fontSize: 10, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getIndexColor(int index) {
    switch (index) {
      case 0:
        return Colors.blue;
      case 1:
        return Colors.indigo;
      case 2:
        return Colors.deepPurple;
      default:
        return Colors.grey;
    }
  }
}

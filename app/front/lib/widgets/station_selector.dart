import 'package:flutter/material.dart';
import '../models/station.dart';

class StationSelector extends StatelessWidget {
  final String label;
  final Station? selectedStation;
  final List<Station> stations;
  final ValueChanged<Station?> onChanged;

  const StationSelector({
    super.key,
    required this.label,
    required this.selectedStation,
    required this.stations,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 10,
            color: Colors.grey[600],
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 2),
        Container(
          height: 36,
          padding: const EdgeInsets.symmetric(horizontal: 8),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: Colors.grey[300]!),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<Station>(
              value: selectedStation,
              hint: const Text('Select', style: TextStyle(fontSize: 12)),
              isExpanded: true,
              isDense: true,
              icon: const Icon(Icons.arrow_drop_down, size: 18),
              style: const TextStyle(fontSize: 12, color: Colors.black87),
              items: stations.map((station) {
                return DropdownMenuItem<Station>(
                  value: station,
                  child: Text(station.name, style: const TextStyle(fontSize: 12)),
                );
              }).toList(),
              onChanged: onChanged,
            ),
          ),
        ),
      ],
    );
  }
}

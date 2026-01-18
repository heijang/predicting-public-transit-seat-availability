import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/location_service.dart';

class LocationDisplay extends StatelessWidget {
  const LocationDisplay({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<LocationService>(
      builder: (context, locationService, child) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.9),
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.1),
                blurRadius: 2,
                offset: const Offset(0, 1),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (locationService.isLoading)
                const SizedBox(
                  width: 10,
                  height: 10,
                  child: CircularProgressIndicator(strokeWidth: 1.5),
                )
              else
                Icon(
                  locationService.currentPosition != null
                      ? Icons.location_on
                      : Icons.location_off,
                  size: 12,
                  color: locationService.currentPosition != null
                      ? Colors.blue
                      : Colors.grey,
                ),
              const SizedBox(width: 4),
              Text(
                locationService.locationText,
                style: TextStyle(
                  fontSize: 9,
                  color: Colors.grey[700],
                  fontWeight: FontWeight.w500,
                ),
              ),
              if (!locationService.isLoading &&
                  locationService.currentPosition == null)
                GestureDetector(
                  onTap: () => locationService.getCurrentLocation(),
                  child: const Padding(
                    padding: EdgeInsets.only(left: 2),
                    child: Icon(Icons.refresh, size: 10, color: Colors.blue),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}

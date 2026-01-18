import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';

class LocationService extends ChangeNotifier {
  Position? _currentPosition;
  String _errorMessage = '';
  bool _isLoading = false;

  Position? get currentPosition => _currentPosition;
  String get errorMessage => _errorMessage;
  bool get isLoading => _isLoading;

  String get locationText {
    if (_currentPosition == null) {
      if (_errorMessage.isNotEmpty) {
        return 'Location error';
      }
      return 'Locating...';
    }
    return '${_currentPosition!.latitude.toStringAsFixed(4)}, ${_currentPosition!.longitude.toStringAsFixed(4)}';
  }

  Future<void> getCurrentLocation() async {
    _isLoading = true;
    _errorMessage = '';
    notifyListeners();

    try {
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        _errorMessage = 'Location service disabled';
        _isLoading = false;
        notifyListeners();
        return;
      }

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          _errorMessage = 'Permission denied';
          _isLoading = false;
          notifyListeners();
          return;
        }
      }

      if (permission == LocationPermission.deniedForever) {
        _errorMessage = 'Permission denied permanently';
        _isLoading = false;
        notifyListeners();
        return;
      }

      _currentPosition = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      _errorMessage = '';
    } catch (e) {
      _errorMessage = 'Cannot get location';
    }

    _isLoading = false;
    notifyListeners();
  }
}

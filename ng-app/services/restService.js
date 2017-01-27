var service = angular.module('ISIApp');

service.factory('Image', ['$resource', function ($resource) {
        return  $resource(
            '/api/images/' + ':id',
            {},
            {}
        );
    }]
).factory('Region', ['$resource', function ($resource) {
        return  $resource(
            '/api/images/' + ':id' + '/id_region',
            {},
            {}
        );
    }]
);

export const DataPermissions = {
  VIEW_ALL_INSPECTIONS:        'view list of all inspections',
  VIEW_OWN_INSPECTIONS:        'view list of own inspections',
  VIEW_OWN_DASHBOARD:          'view own inspection dashboard',
  VIEW_OWN_PORTFOLIO:          'view own inspection portfolio list',
  VIEW_OWN_SCHEDULES:          'view list of own inspection schedules',
  VIEW_OWN_MITIGATIVE_ACTIONS: 'view list of own inspection mitigative actions',
  VIEW_ALL_MITIGATIVE_ACTIONS: 'view list of all inspection mitigative actions',
  CONDUCT_INSPECTION:          'conduct an inspection',
  CAN_BE_INSPECTEE:            'can be an inspectee',
  INSPECTION_API_ACCESS:       'inspection api access',
} as const;

export type DataPermission = typeof DataPermissions[keyof typeof DataPermissions];

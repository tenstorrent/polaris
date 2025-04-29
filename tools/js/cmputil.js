perfdata = null;

function gchart_setup(loadopts)
{
  google.charts.load('current', loadopts);
  google.charts.setOnLoadCallback(drawVisualization);
}


function savecsv(tbl, fname, proxyid)
{
    let hdrstr = tbl[0].join();
    let datstr = tbl.slice(1).map(function(obj){
        let rowstr = obj.join();
        return rowstr;
    }).join("\n");
    let csvstr = hdrstr + "\n" + datstr + "\n";
    let proxy = document.getElementById(proxyid);
    proxy.download = fname;
    proxy.href = 'data:application/csv;charset=utf-8,' + encodeURIComponent(csvstr);
    proxy.target = '_blank';
    proxy.click()
}

function type2jstype(tp)
{
    switch (tp) {
        case "int":
        case "float":
            return "number";
        case "bool":
            return "boolean";
        default:
            return "string";
    }
}

function show_specific_columns(cols)
{
    perfdata.cTableChart.setView({'columns': Array.from(cols).sort((a, b) => a-b)});
    perfdata.bind_dashboard();
}


function toggle_catg(val, catg)
{
    let current_columns = new Set(perfdata.cTableChart.getView().columns);
    let new_columns = null;
    let colsets = perfdata.column_sets;
    if (val.checked)
    {
        new_columns = current_columns.union(colsets[catg]);
    }
    else
    {
        new_columns = current_columns.difference(colsets[catg]).union(colsets['key']).union(colsets['always']);
    }
    show_specific_columns(new_columns);
}

function showall()
{
    document.getElementById('toggle_same_cb').checked = true;
    document.getElementById('toggle_comp_cb').checked = true;
    document.getElementById('toggle_mem_cb').checked = true;
    show_specific_columns(perfdata.column_sets['all']);
}

class PerfData
{
    constructor(attr_desc, comparison_data, linkdir)
    {
        this.attr_desc = attr_desc;
        this.comparison_data = comparison_data;
        this.linkdir = linkdir;
        this.attr_status_frequency = {};
        this.is_attr_identical = {};
        this.column_sets = {};
        this.secondary_column_sets = {};
        this.table_column_names = [];
        this.table_rows = [];
        this.colsummary_table_column_names = ['Colname', 'Seq', '#Matches', 'Total', 'Match?'];
        this.colsummary_table_rows = [];
        this.num_jobs = Object.keys(this.comparison_data['elem_status']).length;
        this.update_numeric_attr();
        this.update_attr_status_frequency();
    }

    update_numeric_attr()
    {
        // is_numeric is a "property" of the python structure, hence does NOT get saved
        // Hence, we explicitly set this attribute.
        for (let attr in this.attr_desc)
        {
            let desc = this.attr_desc[attr];
            desc.is_numeric = desc.attrtype == 'int' || desc.attrtype == 'float';
        }
    }
    update_attr_status_frequency()
    {
        let comparison_data = this.comparison_data;
        let num_jobs = this.num_jobs;
        this.attr_status_frequency = {};
        for (let k in comparison_data['elem_status'])
        {
            let value = comparison_data['elem_status'][k];
            for (let attr in value['elem_status'])
            {
                if (this.attr_status_frequency[attr] == undefined)
                {
                    this.attr_status_frequency[attr] = {};
                }
                let status = value['elem_status'][attr]['status'];
                if (this.attr_status_frequency[attr][status] == undefined)
                {
                    this.attr_status_frequency[attr][status] = 0;
                }
                this.attr_status_frequency[attr][status] += 1;
            }
        }
        this.is_attr_identical = {};
        for (let attr in this.attr_status_frequency)
        {
            this.is_attr_identical[attr] = this.attr_status_frequency[attr]['match'] == num_jobs;
        }
    }


    determine_column_sequence_for_attributes()
    {
        let is_attr_identical = this.is_attr_identical;
        let attr_status_frequency = {};
        this.column_sets = {};
        let column_categories = ['all', 'key', 'perf', 'comp', 'mem', 'info', 'misc', 'same', 'diff', 'numeric', 'integer', 'always'];
        this.column_filters = [];
        let secondary_categories = ['ratios', 'diffs'];
        for (let sndx = 0; sndx < column_categories.length; ++sndx )
        {
            this.column_sets[column_categories[sndx]] = new Set();
        }
        for (let sndx = 0; sndx < secondary_categories.length; ++sndx )
        {
            this.secondary_column_sets[secondary_categories[sndx]] = new Set();
        }

        this.table_column_names = ['Sr'];
        this.column_sets['key'].add(0);
        this.column_sets['all'].add(0);

        this.entries_in_sequence = Object.entries(this.attr_desc).sort(function(a, b) {
            return a[1].seq - b[1].seq;
        });
        let entries = this.entries_in_sequence;
        let num_columns = 0;
        let column_numbers_for_attr = {};
        for (let i = 0; i < entries.length; i++) {
            let desc = entries[i][1];
            let is_identical = is_attr_identical[desc.name];
            let num_cols_for_attr = is_identical ? 1 : desc.is_numeric ? 5 : 3;
            let column_number_list = Array.from({length: num_cols_for_attr}, (_, j) => num_columns + 1 + j)
            column_numbers_for_attr[desc.name]  = column_number_list;
            let column_numbers = new Set(column_number_list);
            num_columns += num_cols_for_attr;
            let same_or_diff = is_identical ? 'same' : 'diff';
            this.column_sets[same_or_diff] = this.column_sets[same_or_diff].union(column_numbers);

            if (desc.is_numeric) {
                let numtype = desc.attrtype == 'int' ? 'integer' : 'numeric';
                if (is_identical)
                {
                    this.column_sets[numtype] = this.column_sets[numtype].union(column_numbers);
                }
                else
                {
                    // 0, 1, 3: numeric, 4: float
                    this.column_sets[numtype].add(column_number_list[0]);
                    this.column_sets[numtype].add(column_number_list[1]);
                    this.column_sets[numtype].add(column_number_list[3]);
                    this.column_sets['numeric'].add(column_number_list[4]);  // ratio can be float
                    this.secondary_column_sets['diffs'].add(column_number_list[3]);
                    this.secondary_column_sets['ratios'].add(column_number_list[4]);
                }
            }
            this.column_sets[desc.catg] = this.column_sets[desc.catg].union(column_numbers);
            this.column_sets['all'] = this.column_sets['all'].union(column_numbers);
            let suffix = is_attr_identical[desc.name] ? '' : ''; // &check;
            if (desc.is_filter) {
                let name = desc.name;
                if (is_attr_identical[desc.name]) {
                    name += suffix;
                }
                // It is the label that seems to be used in filters?? TODO
                if (desc.is_numeric) {
                    this.column_filters.push([desc.label, 'NumberRangeFilter', is_attr_identical[desc.name], desc.label]);
                } else {
                    this.column_filters.push([desc.label, 'CategoryFilter', is_attr_identical[desc.name], desc.label]);
                }
            }
            let jstype = type2jstype(desc.attrtype);
            if (is_attr_identical[desc.name]) {
                this.table_column_names.push({'label': desc.label + suffix, 'name': desc.name, 'type': jstype});
            } else {
                this.table_column_names.push({'label': desc.label + '_1', 'name': desc.name + '_1', 'type': jstype});
                if (desc.is_numeric) {
                    this.table_column_names.push({'label': desc.label + '_2', 'name': desc.name + '_2', 'type': jstype});
                    this.table_column_names.push({'label': desc.label + '_result', 'name': desc.name + '_result', 'type': 'boolean'});
                    this.table_column_names.push({'label': desc.label + '_diff', 'name': desc.name + '_diff', 'type': jstype});
                    this.table_column_names.push({'label': desc.label + '_ratio', 'name': desc.name + '_ratio', 'type': jstype});
                } else {
                    this.table_column_names.push({'label': desc.label + '_2', 'name': desc.name + '_2', jstype});
                    this.table_column_names.push({'label': desc.label + '_result', 'name': desc.name + '_result', 'type': 'boolean'});
                }
            }
        }
        if (this.linkdir != "")
        {
            this.table_column_names.push({'label': 'Link', 'name': 'link', 'type': 'string'});
            this.column_sets['all'].add(num_columns + 1);
            this.column_sets['always'].add(num_columns + 1);
        }
        let catg_toggle_checkboxes = ['same', 'comp', 'mem'];
        let catg_toggle_checkboxes_title = {
            'same': 'Identical',
            'comp': 'Compute',
            'mem' : 'Memory'
        };
        for (let i = 0; i < catg_toggle_checkboxes.length; ++i)
        {
            let catg = catg_toggle_checkboxes[i];
            let checkbox = document.getElementById('toggle_' + catg);
            let nn = this.column_sets[catg].difference(this.column_sets['key']).size
            checkbox.innerHTML = `&nbsp;&nbsp;  ${catg_toggle_checkboxes_title[catg]} Columns (${nn})`; // backticks ` for interpolation
        }
    }

    populate_comparison_table()
    {
        this.table_rows = [];
        this.table_rows.push(this.table_column_names);
        let sr = 0;
        for (const [jobname, jobstatus] of Object.entries(this.comparison_data.elem_status))
        {
            sr += 1;
            let row = [sr];
            let elem_status = jobstatus['elem_status'];
            for (let ndx = 0; ndx < this.entries_in_sequence.length; ndx++)
            {
                let col = this.entries_in_sequence[ndx][0];
                let col_status = elem_status[col];
                let col_desc = this.attr_desc[col];
                if (this.is_attr_identical[col]) {
                    row.push(col_status.value1);
                } else {
                    row.push(col_status.value1, col_status.value2, col_status.status == 'match' || col_status.status == 'approx_match');
                    if (col_desc.is_numeric) {
                        row.push(col_status.diff, col_status.ratio);
                    }
                }
            }
            if (this.linkdir != "")
            {
                row.push('<a href="' + this.linkdir + '/' + jobname + '.html' + '" target="_blank">Link</a>');
            }
            this.table_rows.push(row)
            if (row.length != this.table_column_names.length) {
                console.error('Error: ', row.length, this.table_column_names.length);
                window.stop();
            }
        }
    }

    update_column_summary()
    {
        this.colsummary_table_rows = [];
        this.colsummary_table_rows.push(this.colsummary_table_column_names);
        for (const [colname, colmatches] of Object.entries(this.attr_status_frequency)) {
            let coldesc = this.attr_desc[colname];
            if (! coldesc)
                continue;
            let num_matches = colmatches['match'] || 0;
            num_matches += colmatches['approx_match'] || 0;
            this.colsummary_table_rows.push([colname, coldesc.seq, num_matches, this.num_jobs, num_matches == this.num_jobs])
        }
    }

    setup_gchart_options()
    {
        this.tblcolSummaryOptions = {
            'allowHtml'     : true,
            'showRowNumber' : true,
            'page'          : 'enable',
            'pageSize'      : 5,
            'frozenColumns' : 4,
        };

        this.tblChartOptions = {
            'allowHtml'     : true,
            'showRowNumber' : false,
            'page'          : 'enable',
            'pageSize'      : 10,
            'frozenColumns' : this.column_sets['key'].size,
        };
        this.uioptions = {
            'labelStacking'       : 'vertical',
            'selectedValuesLayout': 'belowStacked'
        };
        this.uioptions_ratio = {
            'labelStacking'       : 'vertical',
            'selectedValuesLayout': 'belowStacked',
            'unitIncrement'       : 0.05,
            'blockIncrement'      : 0.1,
        };
        this.percentFormat    = new google.visualization.NumberFormat({fractionDigits: 2, suffix: "%"});
        this.noDecimalFormat  = new google.visualization.NumberFormat({fractionDigits: 0});
        this.twoDecimalFormat = new google.visualization.NumberFormat({fractionDigits: 2});
        this.highlightFormat  = new google.visualization.ColorFormat();
        this.highlightFormat.addRange(1-epsilon,1+epsilon,'white','green');
        this.diffTwoDecimalFormat = new google.visualization.NumberFormat({negativeColor: 'red', fractionDigits: 2});
    }

    setup_filters()
    {
        let filters = this.column_filters;
        let filter_divs_in_effect = Array();
        if (filters.length == 0)
        {
            console.log('No filter attributes. The dashboard can not be shown.');
            window.stop();
        }
        for(let i=0; i < filters.length; i++){
            let filtertype = filters[i][1];
            let filter_value_identical = filters[i][2];

            filter_divs_in_effect.push('filter-' + i + '-div')
            if (filtertype == 'CategoryFilter' && ! filter_value_identical) {
                filter_divs_in_effect.push('filter-' + i + '-1-div');
            }
        }
        let filterstr = filter_divs_in_effect.map(function(divname){
            return '<div class="col-md-3" id="' + divname + '"></div>';
        }).join('\n');
        filterstr = '<div class="row">\n' + filterstr + '\n</div>\n';
        document.getElementById('abs-stat-filter-div').innerHTML = filterstr;
        this.filterObjs = [];
        for(let i=0; i < filters.length; i++){
            let origcolname    = filters[i][0];
            let filtertype = filters[i][1];
            let filter_value_identical = filters[i][2];
            let colname_label = filters[i][3];
            let options    = this.uioptions;
            let colname = null;
            if(filtertype == 'NumberRangeFilter' && ! filter_value_identical){
                colname = origcolname + '_ratio';
                options = this.uioptions_ratio;
            }else{
                colname = filter_value_identical ? origcolname : origcolname + '_1';
                options = this.uioptions;
            }
            options = JSON.parse(JSON.stringify(options));
            options.label = colname; // colname_label;
            let filterdiv  = 'filter-' + i + '-div';
            this.filterObjs.push(new google.visualization.ControlWrapper({
                'controlType': filtertype,
                'containerId': filterdiv,
                'options': {
                    'filterColumnLabel': colname,
                    'ui': options
                }}));
            if (filtertype != 'NumberRangeFilter' && ! filter_value_identical) {
                let colname = origcolname + '_result';
                options = this.uioptions;
                let filterdiv  = 'filter-' + i + '-1-div';
                this.filterObjs.push(new google.visualization.ControlWrapper({
                    'controlType': filtertype,
                    'containerId': filterdiv,
                    'options': {
                        'filterColumnLabel': colname,
                        'ui': this.uioptions
                    }}));
            }
        }

    }


    setup_dashboard()
    {
        this.sDashBoard  = new google.visualization.Dashboard(document.getElementById('comparison-summary-dashboard-div'));
        this.sTableChart = new google.visualization.ChartWrapper({
            'chartType'  : 'Table',
            'containerId': 'comparison-summary-tbl-div',
            'options'    : this.tblcolSummaryOptions
        });
        this.sFilter = new google.visualization.ControlWrapper({
            'controlType': 'CategoryFilter',
            'containerId': 'comparison-summary-filter-div',
            'options'    : { 'filterColumnLabel': 'Colname', 'ui': this.uioptions }
        });
        document.getElementById('colpagesize-dropdown').addEventListener('change', function() {
            thius.tblcolSummaryOptions.setOption('pageSize', parseInt(this.value));
            this.sTableChart.draw();
        });

        this.summaryDataTable  = google.visualization.arrayToDataTable(this.colsummary_table_rows,false);
        this.summaryDataTable.sort([{'column': 1, 'desc': false}]);
        this.sDashBoard.bind( [this.sFilter], this.sTableChart).draw(this.summaryDataTable);

        this.cDashBoard  = new google.visualization.Dashboard(document.getElementById('abs-stat-dashboard-div'));
        this.cTableChart = new google.visualization.ChartWrapper({
            'chartType'  : 'Table',
            'containerId': 'abs-stat-tbl-div',
            'options'    : this.tblChartOptions
        });
        this.absDataTable  = google.visualization.arrayToDataTable(this.table_rows, false);
        let htmltab = document.getElementById('filter-row-status-table');
        htmltab.rows[0].cells[0].innerHTML = this.table_rows.length-1;
        htmltab.rows[0].cells[2].innerHTML = this.table_rows.length-1;

        let self = this;  // To be used in the callback functions
        // The this passed as the second argument to the map, is the context = "this" for the function being called by map
        Array.from(this.column_sets.numeric).map(function(c) { self.twoDecimalFormat.format(self.absDataTable, c) });
        Array.from(this.column_sets.integer).map(function(c) { self.noDecimalFormat.format(self.absDataTable, c) });
        Array.from(this.secondary_column_sets.ratios).map(function(c)    { self.highlightFormat.format(self.absDataTable, c)});
        Array.from(this.secondary_column_sets.diffs).map(function(c)    { self.diffTwoDecimalFormat.format(self.absDataTable, c)});
        google.visualization.events.addListener(this.cTableChart, 'ready', function() {
            let htmltab = document.getElementById('filter-row-status-table');
            htmltab.rows[0].cells[0].innerHTML = self.cTableChart.getDataTable().getNumberOfRows();
        });
        this.cTableChart.setView({'columns': Array.from(this.column_sets['all'])});
    }

    bind_dashboard()
    {
        let htmltab2 = document.getElementById('filter-column-status-table');
        htmltab2.rows[0].cells[0].innerHTML = this.cTableChart.getView().columns.length;
        htmltab2.rows[0].cells[2].innerHTML = this.column_sets['all'].size;
        this.cDashBoard.bind(this.filterObjs, this.cTableChart ).draw(this.absDataTable);
    }

}

function refreshVisualization()
{
}


function drawVisualization()
{
    perfdata = new PerfData(attr_desc, comparison_data, linkdir);

    perfdata.determine_column_sequence_for_attributes();
    perfdata.populate_comparison_table();
    perfdata.update_column_summary();
    perfdata.setup_gchart_options();
    perfdata.setup_filters();
    perfdata.setup_dashboard();
    perfdata.bind_dashboard();


    /* TODO: Make sure this works even if no filters are specified by users
       i.e. filterObjs array is empty */

    /* TODO: GroupBy support
    let grpDataTable = new google.visualization.data.group(absDataTable,[{ {grp_col} }],[
            {column: { {grp1_col} }, aggregation:google.visualization.data.sum, type:'number'},
            {column: { {grp2_col} }, aggregation:google.visualization.data.sum, type:'number'},
    ]);

    [0,1].map(function(c){ noDecimalFormat.format(grpDataTable,c);});-->
    let xDashBoard  = new google.visualization.Dashboard(document.getElementById('abs-stat-summary-dashboard-div'));
    let xTableChart = new google.visualization.ChartWrapper({
        'chartType'  : 'Table',
        'containerId': 'abs-stat-summary-tbl-div',
        'options'    : tblChartOptions
    });
    let xFilter = new google.visualization.ControlWrapper({
        'controlType': 'CategoryFilter',
        'containerId': 'abs-stat-summary-title-div',
        'options'    : { 'filterColumnLabel': { {filtercolname} }, 'ui': uioptions }
    });
    xDashBoard.bind( [ xFilter ], xTableChart ).draw(grpDataTable);
    */

  }
